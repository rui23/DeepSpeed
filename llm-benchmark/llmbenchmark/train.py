import os
import json
import argparse
import numpy as np

import torch
import transformers
from transformers import (
    set_seed,
    Seq2SeqTrainer, # 继承自Trainer，已经实现了deepseed包裹
)
import deepspeed

from .arguments import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    GenerationArguments,
)
from .callbacks import (
    MMLUEvalCallback,
    EmptycacheCallback,
    PT_ProfCallback,
    StepInfoCallback,
)
from .dataset import (
    make_data_module,
    generate_mmlu_dataset,
)
from .model import (
    get_accelerate_model,
    get_last_checkpoint,
    print_trainable_parameters,
)
from .utils import (
    print_rank_0,
    safe_dict2file,
    get_unique_key,
    hardware_info,
)

def train():    
    # 创建HfArgumentParser对象，用于解析命令行参数
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    # 解析命令行参数并返回数据类对象
    model_args, data_args, training_args, generation_args, extra_args = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    # 将GenerationArguments转换为GenerationConfig对象，并赋值给training_args的generation_config属性
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    # 将所有参数合并到一个args对象中
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    # 打印参数信息
    print_rank_0('打印args信息: ',args)

    # 设置随机种子
    set_seed(args.seed)
    # 获取硬件信息
    hardware = hardware_info()
    n_gpus = hardware.n_gpus
    
    # 获取最后一个检查点的路径和训练是否已完成的标志
    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print_rank_0('Detected that training was already completed!')

    # 获取模型和分词器
    model, tokenizer = get_accelerate_model(args, checkpoint_dir)
    # 禁用缓存
    model.config.use_cache = False
    print_rank_0('model loaded')
    print_rank_0(model)

    # 创建数据模块
    data_module = make_data_module(tokenizer=tokenizer, args=args)

    # 如果不使用硬填充，则抛出异常
    if not args.hard_padding:
        raise ValueError(f"--hard_padding must be True, or throughput may be incorrect.")
    
    # 计算每步的标记数
    token_per_step = args.per_device_train_batch_size * n_gpus * (args.source_max_len + args.target_max_len)

    # 创建Seq2SeqTrainer对象
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )
    
    try:
        print_rank_0(model.hf_device_map)
    except:
        print_rank_0("model has no hf_device_map.")

    # 添加回调函数
    if args.use_lora:
        pass
        # trainer.add_callback(SavePeftModelCallback)

    if args.do_mmlu_eval:
        # 生成MMLU数据集
        mmlu_dataset = generate_mmlu_dataset(args=args)
        # 添加MMLUEvalCallback回调函数
        trainer.add_callback(MMLUEvalCallback(args=args,key=get_unique_key(args),trainer=trainer,tokenizer=tokenizer,mmlu_dataset=mmlu_dataset))
        
    if args.clean_cache:
        # 添加EmptycacheCallback回调函数
        trainer.add_callback(EmptycacheCallback)
        
    # 添加StepInfoCallback回调函数
    trainer.add_callback(StepInfoCallback(warmup_step=args.profiler_warmup_step, key=get_unique_key(args), token_per_step=token_per_step,output_dir=args.output_dir))

    if args.profiler=="deepspeed":
        return NotImplementedError("deepspeed is not supported")
    if args.profiler=="pytorch":
        # 添加PT_ProfCallback回调函数
        trainer.add_callback(PT_ProfCallback(warmup_step=args.profiler_warmup_step, key=get_unique_key(args),output_dir=args.output_dir))
        
    # 打印可训练参数
    print_trainable_parameters(model)

    # 存储所有指标的字典
    all_metrics = {"run_name": args.run_name}
    
    print_rank_0("========START TRAIN========\n")
    if args.do_train:
        print_rank_0("*** Train ***")
        # 执行训练
        train_result = trainer.train()
        metrics = train_result.metrics
        # 记录训练指标
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    if args.do_eval:
        print_rank_0("*** Evaluate ***")
        # 执行评估
        metrics = trainer.evaluate(metric_key_prefix="eval")
        # 记录评估指标
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    if args.do_predict:
        print_rank_0("*** Predict ***")
        # 执行预测
        prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'],metric_key_prefix="predict")
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
            for i, example in enumerate(data_module['predict_dataset']):
                example['prediction_with_input'] = predictions[i].strip()
                example['prediction'] = predictions[i].replace(example['input'], '').strip()
                fout.write(json.dumps(example) + '\n')
        print_rank_0(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))
