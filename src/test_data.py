import logging
import webdataset as wds
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True

def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples

def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    error = 0
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        if not filesample:
            error += 1
            print("filesample is none!!!")
            yield error
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample

input_shards = '/lpai/dataset/cc12m/0-1-0/cc12m-wds/cc12m-train-{0000..2175}.tar'
shard_list = wds.SimpleShardList(input_shards)
dataset = wds.DataPipeline(shard_list)
# for shard in dataset:
#     print(shard)
    
streams = url_opener(dataset, handler=log_and_continue)
files = tar_file_expander(streams, handler=log_and_continue)
for filesample in files:
    print(filesample.keys())
    # if not filesample:
    #     print("filesample is none!!!")
    #     break
# samples = group_by_keys_nothrow(files, handler=log_and_continue)

# samples = tarfile_to_samples_nothrow(dataset)

# dataloader = wds.WebLoader(
#     dataset,
#     batch_size=None,
#     shuffle=False,
#     num_workers=1,  # 单进程模式，方便查看输出
#     persistent_workers=False,
# )
    