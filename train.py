import pandas as pd
import os
from ludwig.api import LudwigModel

original_df = pd.read_csv("train-src-agency-bureau-account-narratives.csv")

# Since many columns are space-padded, remove all padded spaces
training_source_df = original_df.apply(
    lambda x: x.str.strip() if x.dtype == "object" else x
)

# Split columns that have codes embedded
budget_function = training_source_df["BUD_FUNCTION"].str.split(" - ", n=1, expand=True)
training_source_df["BUD_FUNCTION_CODE"] = budget_function[0]
training_source_df["BUD_FUNCTION"] = budget_function[1]

budget_subfn = training_source_df["BUD_SUBFCT"].str.split(" - ", n=1, expand=True)
training_source_df["BUD_SUBFCT_CODE"] = budget_subfn[0]
training_source_df["BUD_SUBFCT"] = budget_subfn[1]

account = training_source_df["ACCOUNT_TITLE"].str.split("   ", n=1, expand=True)
training_source_df["ACCOUNT_CODE"] = account[0]
training_source_df["ACCOUNT_TITLE"] = account[1]

models_common_output_features = [
    # {"name": "BUD_FUNCTION", "type": "category"},
    # {"name": "BUD_SUBFCT", "type": "category"},
    # {"name": "ACCOUNT_TITLE", "type": "category"},
    {"name": "AGENCY_TITLE", "type": "category"},
]

models_defns = {
    "legislative-text-only": LudwigModel(
        model_definition={
            "input_features": [{"name": "PA_NARRATIVE", "type": "text"},],
            "output_features": models_common_output_features,
        }
    ),
    # "pres_budget-text-only": LudwigModel(
    #     model_definition={
    #         "input_features": [{"name": "PN_NARRATIVE", "type": "text"},],
    #         "output_features": models_common_output_features,
    #     }
    # ),
    # "all-text": LudwigModel(
    #     model_definition={
    #         "input_features": [
    #             {"name": "PA_NARRATIVE", "type": "text"},
    #             {"name": "PN_NARRATIVE", "type": "text"},
    #         ],
    #         "output_features": models_common_output_features,
    #     }
    # ),
}

models_trained_path = "./models-trained"
os.mkdir(models_trained_path)

for model_name, ludwig_model in models_defns.items():
    training_results = ludwig_model.train(data_df=training_source_df)
    ludwig_model.save(os.path.join(models_trained_path, model_name))
    ludwig_model.close()
