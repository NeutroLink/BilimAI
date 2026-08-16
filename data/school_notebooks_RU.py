import os
import json
import datasets


class SchoolNotebooks(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                }
            )
        )

    def _split_generators(self, dl_manager):
        _URLS = {
            "images": "images.zip",
            "train_data": "annotations_train.json",
            "test_data": "annotations_test.json",
            "val_data": "annotations_val.json"
        }
        data_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "image_paths": dl_manager.iter_files(data_files["images"]),
                    "annotation_path": data_files["train_data"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "image_paths": dl_manager.iter_files(data_files["images"]),
                    "annotation_path": data_files["test_data"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "image_paths": dl_manager.iter_files(data_files["images"]),
                    "annotation_path": data_files["val_data"],
                },
            )
        ]

    def _generate_examples(self, image_paths, annotation_path):
        """Generate examples."""
        with open(annotation_path, 'r') as f:
            data = json.load(f)

        image_names = set()
        for image_data in data['images']:
            image_names.add(image_data['file_name'])

        for idx, image_path in enumerate(image_paths):
            if os.path.basename(image_path) in image_names:
                example = {
                    "image": image_path,
                }
                yield idx, example
