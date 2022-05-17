import logging
import os
from typing import Optional, Union

from transformers import AutoConfig
from transformers.modeling_utils import (
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    EntryNotFoundError,
    HTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    cached_path,
    get_checkpoint_shard_files,
    hf_bucket_url,
    is_remote_url,
    load_state_dict,
)

from corener.utils import set_logger

set_logger()

WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"

"""Modification of the code here: 
https://github.com/huggingface/transformers/blob/v4.19.0/src/transformers/modeling_utils.py#L1560"""


def load_weights_and_config(
    pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs
):
    state_dict = kwargs.pop("state_dict", None)
    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", False)
    use_auth_token = kwargs.pop("use_auth_token", None)
    revision = kwargs.pop("revision", None)
    mirror = kwargs.pop("mirror", None)
    from_pipeline = kwargs.pop("_from_pipeline", None)
    from_auto_class = kwargs.pop("_from_auto", False)
    _fast_init = kwargs.pop("_fast_init", True)
    torch_dtype = kwargs.pop("torch_dtype", None)
    low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)

    user_agent = {
        "file_type": "model",
        "framework": "pytorch",
        "from_auto_class": from_auto_class,
    }
    if from_pipeline is not None:
        user_agent["using_pipeline"] = from_pipeline

    # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
    # index of the files.
    is_sharded = False
    sharded_metadata = None
    # Load model
    if pretrained_model_name_or_path is not None:
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            if os.path.isfile(
                os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
            ):
                # Load from a PyTorch checkpoint
                archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
            elif os.path.isfile(
                os.path.join(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME)
            ):
                # Load from a sharded PyTorch checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, WEIGHTS_INDEX_NAME
                )
                is_sharded = True
            else:
                raise EnvironmentError(
                    f"Error no file named {WEIGHTS_NAME} found in directory {pretrained_model_name_or_path}."
                )

        elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(
            pretrained_model_name_or_path
        ):
            archive_file = pretrained_model_name_or_path

        else:
            # set correct filename
            filename = WEIGHTS_NAME

            archive_file = hf_bucket_url(
                pretrained_model_name_or_path,
                filename=filename,
                revision=revision,
                mirror=mirror,
            )

        try:
            # Load from URL or cache if already cached
            resolved_archive_file = cached_path(
                archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
            )

        except RepositoryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier "
                "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a "
                "token having permission to this repo with `use_auth_token` or log in with `huggingface-cli "
                "login` and pass `use_auth_token=True`."
            )
        except RevisionNotFoundError:
            raise EnvironmentError(
                f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for "
                "this model name. Check the model page at "
                f"'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
            )
        except EntryNotFoundError:
            if filename == WEIGHTS_NAME:
                try:
                    # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                    archive_file = hf_bucket_url(
                        pretrained_model_name_or_path,
                        filename=WEIGHTS_INDEX_NAME,
                        revision=revision,
                        mirror=mirror,
                    )
                    resolved_archive_file = cached_path(
                        archive_file,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        user_agent=user_agent,
                    )
                    is_sharded = True
                except EntryNotFoundError:
                    raise EnvironmentError(
                        f"{pretrained_model_name_or_path} does not appear to have a file named {WEIGHTS_NAME}."
                    )
            else:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} does not appear to have a file named {filename}."
                )
        except HTTPError as err:
            raise EnvironmentError(
                f"There was a specific connection error when trying to load {pretrained_model_name_or_path}:\n"
                f"{err}"
            )
        except ValueError:
            raise EnvironmentError(
                f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it in the cached "
                f"files and it looks like {pretrained_model_name_or_path} is not the path to a directory "
                f"containing a file named {WEIGHTS_NAME}.\n"
                "Checkout your internet connection or see how to run the library in offline mode at "
                "'https://huggingface.co/docs/transformers/installation#offline-mode'."
            )
        except EnvironmentError:
            raise EnvironmentError(
                f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                f"containing a file named {WEIGHTS_NAME}."
            )

        if resolved_archive_file == archive_file:
            logging.info(f"loading weights file {archive_file}")
        else:
            logging.info(
                f"loading weights file {archive_file} from cache at {resolved_archive_file}"
            )
    else:
        resolved_archive_file = None

    # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
    if is_sharded:
        # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            resolved_archive_file,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            user_agent=user_agent,
            revision=revision,
            mirror=mirror,
        )

    # load pt weights early so that we know which dtype to init the model under
    if not is_sharded and state_dict is None:
        # Time to load the checkpoint
        state_dict = load_state_dict(resolved_archive_file)

    if is_sharded:
        loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
    else:
        loaded_state_dict_keys = [k for k in state_dict.keys()]
    if low_cpu_mem_usage:
        state_dict = None

    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    return state_dict, loaded_state_dict_keys, config
