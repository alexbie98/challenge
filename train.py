import argparse
import numpy as np

import datasets
import transformers

import lib.data



def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=2e-4)
    args = parser.parse_args()


    # loading the dataset
    print('| loading dataset...')
    ds = datasets.load_dataset('beans', cache_dir='data')
    print(f'| num train examples: {len(ds["train"])}')
    print(f'| num test examples: {len(ds["test"])}')

    labels = ds['train'].features['labels'].names
    print(f'| labels: {labels}')

    # load pretrained model
    model_path = 'google/vit-base-patch16-224-in21k'
    feature_extractor = transformers.ViTFeatureExtractor.from_pretrained(model_path)

    # add transform
    prepared_ds = ds.with_transform(lambda x: lib.data.transform(x, feature_extractor))


    # metrics
    metric = datasets.load_metric("accuracy")
    def compute_metrics(p):
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    # model
    model = transformers.ViTForImageClassification.from_pretrained(
        model_path,
        num_labels = len(labels),
        id2label = {str(i): c for i,c in enumerate(labels)},
        label2id = {c: str(i) for i,c in enumerate(labels)},
    )

    # run training
    training_args = transformers.TrainingArguments(
        output_dir="results",
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=1,
        fp16=False,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=args.lr,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to='tensorboard',
        load_best_model_at_end=True,
    )

    trainer = transformers.Trainer(
        model = model,
        args = training_args,
        data_collator=lib.data.collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_ds['train'],
        eval_dataset=prepared_ds['validation'],
        tokenizer=feature_extractor
    )


    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    # run eval
    metrics = trainer.evaluate(prepared_ds['validation'])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__ == '__main__':
    main()
