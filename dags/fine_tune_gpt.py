"""
## Fine-tune GPT-3-turbo using custom examples

Fine-tunes GPT-3 turbo on the latest train and validation examples. 
Manual runs offer the option to choose the specific example files.
"""

from airflow.decorators import dag, task
from airflow.models.dataset import Dataset
from airflow.models.param import Param
from airflow.models.baseoperator import chain
from pendulum import datetime, duration
import logging
import os

t_log = logging.getLogger("airflow.task")

_COMBINED_TRAIN_EXAMPLES_URI = os.getenv("COMBINED_TRAIN_EXAMPLES_URI")
_COMBINED_VALIDATION_EXAMPLES_URI = os.getenv("COMBINED_VALIDATION_EXAMPLES_URI")
_CHALLENGER_MODEL_INFO_URI = os.getenv("CHALLENGER_MODEL_INFO_URI")

from include.custom_operators.gpt_fine_tune import OpenAIFineTuneOperator


@dag(
    dag_display_name="🤖 Fine-tune",
    start_date=datetime(2024, 4, 1),
    schedule=(
        Dataset(_COMBINED_TRAIN_EXAMPLES_URI)
        & Dataset(_COMBINED_VALIDATION_EXAMPLES_URI)
    ),
    catchup=False,
    max_consecutive_failed_dag_runs=5,
    render_template_as_native_obj=True,
    tags=["GPT-3.5-turbo", "use-case"],
    default_args={
        "retries": 3,
        "retry_delay": duration(minutes=5),
        "retry_exponential_backoff": True,
        "owner": "MLE team",
    },
    params={
        "manual_choice_train_examples_file_path": Param(
            None,
            type=["null", "string"],
            description="Path to the file containing examples for fine-tuning. Example: `include/examples/train_examples/fine_tune_examples/20240417T125742_combined_examples.json`",
        ),
        "manual_choice_validation_examples_file_path": Param(
            None,
            type=["null", "string"],
            description="Path to the file containing examples for fine-tuning. Example: `include/examples/validation_examples/fine_tune_examples/20240417T125742_combined_examples.json`",
        ),
        "manual_model_fine_tuning_suffix": Param(
            None,
            type=["null", "string"],
            description="Optional. Manually set the suffix for the fine-tuning job. If not set, the suffix will be generated automatically based on the DAG run id. Max 18 characters.",
            maxLength=18,
        ),
    },
    doc_md=__doc__,
    description="Fine tune GPT-3.5-turbo.",
)
def fine_tune_gpt():

    @task
    def choose_examples_file(type: str, **context) -> str:
        """
        Determine which examples file to use.
        Returns:
            str: Path to the chosen examples file.
        """

        if context["params"][f"manual_choice_{type}_examples_file_path"]:
            examples_file_path = context["params"][
                f"manual_choice_{type}_examples_file_path"
            ]

        else:
            examples_file_path = context["ti"].xcom_pull(
                task_ids=f"create_fine_tuning_{type}_example_file",
                dag_id="eval_and_cost_estimate",
                include_prior_dates=True,
            )
            print(examples_file_path)

        return examples_file_path

    @task
    def upload_examples_file_to_openai(examples_file_path: str) -> str:
        """
        Uploads an examples file to OpenAI.
        Args:
            examples_file_path (str): Path to the examples file.
        Returns:
            str: File ID of the uploaded examples file.
        """

        from openai import OpenAI

        client = OpenAI()  # fetches the API key from the environment var OPENAI_API_KEY
        r = client.files.create(
            file=open(examples_file_path, "rb"), purpose="fine-tune"
        )

        t_log.info(f"Uploaded examples file to OpenAI. File ID: {r.id}")
        t_log.info(f"File name: {r.filename}")
        t_log.info(f"File size: {round(r.bytes / 1000, 2)} KB")

        return r.id

    upload_train_examples_file_to_openai_obj = upload_examples_file_to_openai.override(
        task_id="upload_train_examples_file_to_openai"
    )(
        examples_file_path=choose_examples_file.override(
            task_id="choose_train_examples_file"
        )(type="train")
    )

    upload_validation_examples_file_to_openai_obj = (
        upload_examples_file_to_openai.override(
            task_id="upload_validation_examples_file_to_openai"
        )(
            examples_file_path=choose_examples_file.override(
                task_id="choose_validation_examples_file"
            )(type="validation")
        )
    )

    fine_tune = OpenAIFineTuneOperator(
        task_id="fine_tune",
        fine_tuning_file_id=upload_train_examples_file_to_openai_obj,
        validation_file_id=upload_validation_examples_file_to_openai_obj,
        model="gpt-3.5-turbo",
        suffix="{{ params.manual_model_fine_tuning_suffix }}",
        wait_for_completion=True,
        deferrable=True,
        poke_interval=5,
        retries=0,
    )

    @task(outlets=[Dataset(_CHALLENGER_MODEL_INFO_URI)])
    def update_dataset():
        t_log.info("New model fine-tuned successfully. Updating the dataset.")

    chain(fine_tune, update_dataset())


fine_tune_gpt()
