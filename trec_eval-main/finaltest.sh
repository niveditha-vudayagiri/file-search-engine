./trec_eval -q -m all_trec results_testing/cranqrel.test.txt results_testing/vsm_results.trec > results_testing/vsm_out.test
./trec_eval -q -m all_trec results_testing/cranqrel.test.txt results_testing/bm25_results.trec > results_testing/bm_out.test
./trec_eval -q -m all_trec results_testing/cranqrel.test.txt results_testing/lm_results.trec > results_testing/lm_out.test
