import numpy as np

class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, dataset, input_dimension=6, output_dimension=6, width=300):

        if False:
            channels = 256

            return [
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": 1, "kernal": 3, "stride": 2, "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},
                #
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'flatten'},
                # {"name": 'rotate'},
                {"name": 'rep'},

                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": 1000, "in": 9 * channels}}

            ]

        elif dataset == "omniglot":
            channels = 256

            return [
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": 128, "in-channels": 1, "kernal": 4, "stride": 2, "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": 128, "in-channels": 128, "kernal": 4
                     , "stride": 1, "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": 128, "in-channels": 128, "kernal": 4, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": 128, "in-channels": 128, "kernal": 3, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'flatten'},
                # {"name": 'rotate'},
                {"name": 'rep'},

                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": 170, "in": 512}},
                {"name": 'relu'},

                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": 120, "in": 170}},
                {"name": 'relu'},

                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": 1000, "in": 120}}

            ]

        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)
