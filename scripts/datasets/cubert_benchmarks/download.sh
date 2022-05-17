mkdir function_docstring_datasets
gsutil -m cp \
  "gs://cubert/20200621_Python/function_docstring_datasets/dev.jsontxt-00000-of-00004" \
  "gs://cubert/20200621_Python/function_docstring_datasets/dev.jsontxt-00001-of-00004" \
  "gs://cubert/20200621_Python/function_docstring_datasets/dev.jsontxt-00002-of-00004" \
  "gs://cubert/20200621_Python/function_docstring_datasets/dev.jsontxt-00003-of-00004" \
  "gs://cubert/20200621_Python/function_docstring_datasets/eval.jsontxt-00000-of-00004" \
  "gs://cubert/20200621_Python/function_docstring_datasets/eval.jsontxt-00001-of-00004" \
  "gs://cubert/20200621_Python/function_docstring_datasets/eval.jsontxt-00002-of-00004" \
  "gs://cubert/20200621_Python/function_docstring_datasets/eval.jsontxt-00003-of-00004" \
  "gs://cubert/20200621_Python/function_docstring_datasets/train.jsontxt-00000-of-00004" \
  "gs://cubert/20200621_Python/function_docstring_datasets/train.jsontxt-00001-of-00004" \
  "gs://cubert/20200621_Python/function_docstring_datasets/train.jsontxt-00002-of-00004" \
  "gs://cubert/20200621_Python/function_docstring_datasets/train.jsontxt-00003-of-00004" \
  function_docstring_datasets

mkdir exception_datasets
gsutil -m cp \
  "gs://cubert/20200621_Python/exception_datasets/dev.jsontxt-00000-of-00004" \
  "gs://cubert/20200621_Python/exception_datasets/dev.jsontxt-00001-of-00004" \
  "gs://cubert/20200621_Python/exception_datasets/dev.jsontxt-00002-of-00004" \
  "gs://cubert/20200621_Python/exception_datasets/dev.jsontxt-00003-of-00004" \
  "gs://cubert/20200621_Python/exception_datasets/eval.jsontxt-00000-of-00004" \
  "gs://cubert/20200621_Python/exception_datasets/eval.jsontxt-00001-of-00004" \
  "gs://cubert/20200621_Python/exception_datasets/eval.jsontxt-00002-of-00004" \
  "gs://cubert/20200621_Python/exception_datasets/eval.jsontxt-00003-of-00004" \
  "gs://cubert/20200621_Python/exception_datasets/train.jsontxt-00000-of-00004" \
  "gs://cubert/20200621_Python/exception_datasets/train.jsontxt-00001-of-00004" \
  "gs://cubert/20200621_Python/exception_datasets/train.jsontxt-00002-of-00004" \
  "gs://cubert/20200621_Python/exception_datasets/train.jsontxt-00003-of-00004" \
  exception_datasets

mkdir variable_misuse_datasets
gsutil -m cp \
  "gs://cubert/20200621_Python/variable_misuse_datasets/dev.jsontxt-00000-of-00004" \
  "gs://cubert/20200621_Python/variable_misuse_datasets/dev.jsontxt-00001-of-00004" \
  "gs://cubert/20200621_Python/variable_misuse_datasets/dev.jsontxt-00002-of-00004" \
  "gs://cubert/20200621_Python/variable_misuse_datasets/dev.jsontxt-00003-of-00004" \
  "gs://cubert/20200621_Python/variable_misuse_datasets/eval.jsontxt-00000-of-00004" \
  "gs://cubert/20200621_Python/variable_misuse_datasets/eval.jsontxt-00001-of-00004" \
  "gs://cubert/20200621_Python/variable_misuse_datasets/eval.jsontxt-00002-of-00004" \
  "gs://cubert/20200621_Python/variable_misuse_datasets/eval.jsontxt-00003-of-00004" \
  "gs://cubert/20200621_Python/variable_misuse_datasets/train.jsontxt-00000-of-00004" \
  "gs://cubert/20200621_Python/variable_misuse_datasets/train.jsontxt-00001-of-00004" \
  "gs://cubert/20200621_Python/variable_misuse_datasets/train.jsontxt-00002-of-00004" \
  "gs://cubert/20200621_Python/variable_misuse_datasets/train.jsontxt-00003-of-00004" \
  variable_misuse_datasets

mkdir swapped_operands_datasets
gsutil -m cp \
  "gs://cubert/20200621_Python/swapped_operands_datasets/dev.jsontxt-00000-of-00004" \
  "gs://cubert/20200621_Python/swapped_operands_datasets/dev.jsontxt-00001-of-00004" \
  "gs://cubert/20200621_Python/swapped_operands_datasets/dev.jsontxt-00002-of-00004" \
  "gs://cubert/20200621_Python/swapped_operands_datasets/dev.jsontxt-00003-of-00004" \
  "gs://cubert/20200621_Python/swapped_operands_datasets/eval.jsontxt-00000-of-00004" \
  "gs://cubert/20200621_Python/swapped_operands_datasets/eval.jsontxt-00001-of-00004" \
  "gs://cubert/20200621_Python/swapped_operands_datasets/eval.jsontxt-00002-of-00004" \
  "gs://cubert/20200621_Python/swapped_operands_datasets/eval.jsontxt-00003-of-00004" \
  "gs://cubert/20200621_Python/swapped_operands_datasets/train.jsontxt-00000-of-00004" \
  "gs://cubert/20200621_Python/swapped_operands_datasets/train.jsontxt-00001-of-00004" \
  "gs://cubert/20200621_Python/swapped_operands_datasets/train.jsontxt-00002-of-00004" \
  "gs://cubert/20200621_Python/swapped_operands_datasets/train.jsontxt-00003-of-00004" \
  swapped_operands_datasets

mkdir wrong_binary_operator_datasets
gsutil -m cp \
  "gs://cubert/20200621_Python/wrong_binary_operator_datasets/dev.jsontxt-00000-of-00004" \
  "gs://cubert/20200621_Python/wrong_binary_operator_datasets/dev.jsontxt-00001-of-00004" \
  "gs://cubert/20200621_Python/wrong_binary_operator_datasets/dev.jsontxt-00002-of-00004" \
  "gs://cubert/20200621_Python/wrong_binary_operator_datasets/dev.jsontxt-00003-of-00004" \
  "gs://cubert/20200621_Python/wrong_binary_operator_datasets/eval.jsontxt-00000-of-00004" \
  "gs://cubert/20200621_Python/wrong_binary_operator_datasets/eval.jsontxt-00001-of-00004" \
  "gs://cubert/20200621_Python/wrong_binary_operator_datasets/eval.jsontxt-00002-of-00004" \
  "gs://cubert/20200621_Python/wrong_binary_operator_datasets/eval.jsontxt-00003-of-00004" \
  "gs://cubert/20200621_Python/wrong_binary_operator_datasets/train.jsontxt-00000-of-00004" \
  "gs://cubert/20200621_Python/wrong_binary_operator_datasets/train.jsontxt-00001-of-00004" \
  "gs://cubert/20200621_Python/wrong_binary_operator_datasets/train.jsontxt-00002-of-00004" \
  "gs://cubert/20200621_Python/wrong_binary_operator_datasets/train.jsontxt-00003-of-00004" \
  wrong_binary_operator_datasets

mkdir variable_misuse_repair_datasets
gsutil -m cp \
  "gs://cubert/20200621_Python/variable_misuse_repair_datasets/dev.jsontxt-00000-of-00004" \
  "gs://cubert/20200621_Python/variable_misuse_repair_datasets/dev.jsontxt-00001-of-00004" \
  "gs://cubert/20200621_Python/variable_misuse_repair_datasets/dev.jsontxt-00002-of-00004" \
  "gs://cubert/20200621_Python/variable_misuse_repair_datasets/dev.jsontxt-00003-of-00004" \
  "gs://cubert/20200621_Python/variable_misuse_repair_datasets/eval.jsontxt-00000-of-00006" \
  "gs://cubert/20200621_Python/variable_misuse_repair_datasets/eval.jsontxt-00001-of-00006" \
  "gs://cubert/20200621_Python/variable_misuse_repair_datasets/eval.jsontxt-00002-of-00006" \
  "gs://cubert/20200621_Python/variable_misuse_repair_datasets/eval.jsontxt-00003-of-00006" \
  "gs://cubert/20200621_Python/variable_misuse_repair_datasets/eval.jsontxt-00004-of-00006" \
  "gs://cubert/20200621_Python/variable_misuse_repair_datasets/eval.jsontxt-00005-of-00006" \
  "gs://cubert/20200621_Python/variable_misuse_repair_datasets/githubcommits.jsontxt-00000-of-00001" \
  "gs://cubert/20200621_Python/variable_misuse_repair_datasets/githubcommits.raw.jsontxt-00000-of-00001" \
  "gs://cubert/20200621_Python/variable_misuse_repair_datasets/train.jsontxt-00000-of-00011" \
  "gs://cubert/20200621_Python/variable_misuse_repair_datasets/train.jsontxt-00001-of-00011" \
  "gs://cubert/20200621_Python/variable_misuse_repair_datasets/train.jsontxt-00002-of-00011" \
  "gs://cubert/20200621_Python/variable_misuse_repair_datasets/train.jsontxt-00003-of-00011" \
  "gs://cubert/20200621_Python/variable_misuse_repair_datasets/train.jsontxt-00004-of-00011" \
  "gs://cubert/20200621_Python/variable_misuse_repair_datasets/train.jsontxt-00005-of-00011" \
  "gs://cubert/20200621_Python/variable_misuse_repair_datasets/train.jsontxt-00006-of-00011" \
  "gs://cubert/20200621_Python/variable_misuse_repair_datasets/train.jsontxt-00007-of-00011" \
  "gs://cubert/20200621_Python/variable_misuse_repair_datasets/train.jsontxt-00008-of-00011" \
  "gs://cubert/20200621_Python/variable_misuse_repair_datasets/train.jsontxt-00009-of-00011" \
  "gs://cubert/20200621_Python/variable_misuse_repair_datasets/train.jsontxt-00010-of-00011" \
  variable_misuse_repair_datasets