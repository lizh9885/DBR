from datasets import load_dataset


def main():
    raw_dset = load_dataset("glue", "mnli")
    raw_dset = raw_dset.rename_column("premise", "text1")
    raw_dset = raw_dset.rename_column("hypothesis", "text2")
    text_labs = ["entailment", "neutral", "contradiction"]

    for split, dset in raw_dset.items():
        label_text = []
        for i in dset["label"]:
            if i >= 0:
                label_text.append(text_labs[i])
            else:
                label_text.append("unlabeled")
        dset = dset.add_column("label_text", label_text)        
        dset.to_json(f"{split}.jsonl")


if __name__ == "__main__":
    main()
    print("Job's done!")