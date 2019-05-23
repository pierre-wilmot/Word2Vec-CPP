#include <fstream>
#include <iostream>

#include "averaged_embeddings.hpp"
#include "embeddings.hpp"
#include "graph.hpp"
#include "matrix_multiply.hpp"
#include "inplace_transpose.hpp"
#include "placeholder.hpp"
#include "sigmoid.hpp"
#include "tensor.hpp"
#include "w2v_cbow_dataloader.hpp"

using namespace fetch::ml;
using namespace fetch::ml::ops;

#define EMBEDDINGS_SIZE 100
#define NB_EPOCH 10
#define NEGATIVE_SAMPLES 25
#define MINIMUM_WORD_FREQUENCY 5
#define OUTPUT_FILE "vector.bin"

std::string readFile(std::string const &path)
{
  std::ifstream t(path);
  return std::string((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
}

void saveVectors(std::string const &output_file,
		 fetch::math::Tensor<float, 2> const &matrix,
		 std::map<std::string, std::pair<uint64_t, uint64_t>> const &vocab)
{
  std::fstream myfile(output_file, std::ios::out | std::ios::binary);
  myfile << vocab.size() << " " << matrix.shape()[1] << "\n";
  for (auto kvp : vocab)
    {
      myfile << kvp.first << " ";
      for (float const &v : matrix.Slice(kvp.second.first))
	myfile.write((char *)&v, sizeof(float));
      myfile << "\n";
    }
  myfile.close();
}

int main(int ac, char **av)
{
  std::cout << "Word2Vec" << std::endl;
  if (ac < 2)
    {
      std::cerr << "Usage : " << av[0] << " CORPUS_FILES ..." << std::endl;
      return 1;
    }

  // Loading the text data
  fetch::ml::CBOWLoader<float> loader(5, NEGATIVE_SAMPLES);
  for (int i(1) ; i < ac ; ++i)
    loader.AddData(readFile(av[i]));
  loader.RemoveInfrequent(MINIMUM_WORD_FREQUENCY);
  loader.InitUnigramTable();
  std::cout << "Vocab size : " << loader.VocabSize() << std::endl;

  // Allocating and initialising the matrix that contains word vectors
  fetch::math::Tensor<float, 2> word_embeding_matrix = fetch::math::Tensor<float, 2>({loader.VocabSize(), EMBEDDINGS_SIZE});
  for (auto &e : word_embeding_matrix)
    e = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) / EMBEDDINGS_SIZE;

  // Setting up the graph
  Graph<fetch::math::Tensor<float, 2>> graph;
  graph.AddNode<PlaceHolder<fetch::math::Tensor<float, 2>, 2>>("Context", {});
  graph.AddNode<AveragedEmbeddings<fetch::math::Tensor<float, 2>>>("Words", {"Context"}, word_embeding_matrix);
  graph.AddNode<PlaceHolder<fetch::math::Tensor<float, 2>, 2>>("Target", {});
  graph.AddNode<Embeddings<fetch::math::Tensor<float, 2>>>("Weights", {"Target"}, loader.VocabSize(), EMBEDDINGS_SIZE);
  graph.AddNode<InplaceTranspose<fetch::math::Tensor<float, 2>>>("WeightsTranspose", {"Weights"});  
  graph.AddNode<MatrixMultiply<fetch::math::Tensor<float, 2>>>("DotProduct", {"Words", "WeightsTranspose"});
  graph.AddNode<Sigmoid<fetch::math::Tensor<float, 2>>>("Sigmoid", {"DotProduct"});

  // Learning rate
  float initial_learning_rate = 0.05f;
  float learning_rate = initial_learning_rate;
  float minimum_learning_rate = initial_learning_rate * 0.0001;

  // Training loop
  auto sample = loader.GetNext();
  fetch::math::Tensor<float, 2> error({1, 25}); // This buffer store the error
  fetch::math::Tensor<float, 2> ground_truth({1, 25}); // This one the ground truth
  ground_truth.Fill(0); // All negative samples
  ground_truth.Set(0, 0, 1.0f); // Except first one
  unsigned int total_number_iterations = NB_EPOCH * loader.Size();
  unsigned int i(0);
  for (int epoch(0) ; epoch < NB_EPOCH ; ++epoch)
    {
      std::cout << "Epoch " << epoch << std::endl;
      loader.Reset();
      while (!loader.IsDone())
	{
	  // Retrieve next data sample
	  // The data consits of an averged context vector [1x200] (sample.first)
	  //                  and a matrix of [25x200] where :
	  // the first row correspond to the weight vector of the positive sample (the word that was actually part of the corpus)
	  // the 24th others are weight vectors for negatives samples, choosen according to the unigram table
	  loader.GetNext(sample);
	  graph.SetInput("Context", sample.first);
	  graph.SetInput("Target", sample.second);

	  auto const &prediction = graph.Evaluate("Sigmoid");
	  error.Copy(ground_truth);
	  error.InlineSubtract(prediction);
	  // This is not a mistake : the original Google C code does this very strange thing
	  // They clamp the output using a sigmoid, but never actually run the backward pass for it
	  graph.BackPropagate("DotProduct", error);

	  // Adjust the learning rate
	  learning_rate = std::max((static_cast<float>(total_number_iterations - i) / total_number_iterations) * initial_learning_rate, minimum_learning_rate);
	  graph.Step(learning_rate);
	  i++;
	}
    }

  // Saving the trained vectors to disk
  saveVectors(OUTPUT_FILE, word_embeding_matrix, loader.GetVocab());
  
  return 0;
}
