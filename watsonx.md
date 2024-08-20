Use these setting files to connect to watsonx models:
see here for where to put the files: https://aider.chat/docs/config/adv-model-settings.html


.aider.model.metadata.json
```
{
    "watsonx/meta-llama/llama-3-8b-instruct": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0000018,
        "output_cost_per_token": 0.0000018,
        "litellm_provider": "watsonx",
        "mode": "chat",
      "source": "https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html?context=wx&locale=en&audience=wdp#llama-3-1"
    },
    "watsonx/meta-llama/llama-3-1-70b-instruct": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0000018,
        "output_cost_per_token": 0.0000018,
        "litellm_provider": "watsonx",
        "mode": "chat",
      "source": "https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html?context=wx&locale=en&audience=wdp#llama-3-1"
    },
    "watsonx/meta-llama/llama-3-405b-instruct": {
        "max_tokens":  16384,
        "max_input_tokens": 126976,
        "max_output_tokens":  16384,
        "input_cost_per_token": 0.000005,
        "output_cost_per_token": 0.000016,
        "litellm_provider": "watsonx",
        "mode": "chat",
      "source": "https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html?context=wx&locale=en&audience=wdp#llama-3-1"
    }
}
```

.aider.model.settings.yml
```
- name: "watsonx/meta-llama/llama-3-8b-instruct"
  edit_format: "whole"
  weak_model_name: "watsonx/meta-llama/llama-3-8b-instruct"
  use_repo_map: false
  send_undo_reply: false
  accepts_images: false
  lazy: false
  reminder_as_sys_msg: true
  examples_as_sys_msg: false
  max_tokens: 4096
- name: "watsonx/meta-llama/llama-3-1-70b-instruct"
  edit_format: "udiff"
  weak_model_name: "watsonx/meta-llama/llama-3-8b-instruct"
  use_repo_map: true
  send_undo_reply: true
  accepts_images: true
  lazy: true
  reminder_as_sys_msg: true
  examples_as_sys_msg: false
  max_tokens: 4096
- name: "watsonx/meta-llama/llama-3-405b-instruct"
  edit_format: "udiff"
  weak_model_name: "watsonx/meta-llama/llama-3-8b-instruct"
  use_repo_map: true
  send_undo_reply: true
  accepts_images: true
  lazy: true
  reminder_as_sys_msg: true
  examples_as_sys_msg: false
  max_tokens: 16384
```
