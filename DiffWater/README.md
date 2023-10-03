experimental procedure:

1. Prepare the dataset: The training set is split into two paths: input_256 and target_256, and the same is true for the test set.
2. Change the parameter configuration file: Open configs.yml and change the dataset path in it.
3. Change where the pre-trained model is stored: see path.resume_state in configs.yml.

4. Make sure the computer can use cuda. If it only has cpu, it won't run Run train.py and see if it works with the current Settings.

6. Change datasets.train.batch_size in configs.yml according to the situation.

7. Modify the number of times required for training according to actual requirements. The default n_inter is 1000000, see train.n_inter value in configs.yml.

The experiments folder will store the results on the validation dataset during training.

9. Training is over.

10. Note that the test path also has both input _256 and target_256 paths, although target is not involved in the test process.

11. Run test.py.

12. The test is over.

13. Run eval.py directly to calculate the metrics.