# Handwritten-Digit-Recognition-Neural-Network-from-Scratch

This project is based on learning from https://github.com/Maitreya-Bhatkhande/neural-network-from-scratch-numpy. It is a practice project in building a neural network from scratch. The highlight is the insights I gained while working through some of the headaches I encountered along the way. I've also made some improvements and explained important concepts, especially the mathematical logic behind the neural network.

## Structure
1. Code for this specific project (Digit Recognition)
2. Core concepts and my exploration
   - 2.1 Dimensions and matrix operations: What are the shapes of the parameters, activations, and gradients, and what do the matrix operations mean during forward and backward propagation? (Maths, Difficult)
   - 2.2 Learning rate: How should we choose a learning rate, and how does it affect convergence? (Experiment)
   - 2.3 Weight initialization: How can we choose appropriate initial weights for a neural network? (Maths + Experiment)
   - 2.4 Training difficulties: How can we recognize and investigate oscillation caused by an excessive learning rate, as well as overfitting? (Experiment)
   - 2.5 Conclusion: My main takeaways, experimental findings, and next steps.
3. Modified DNN class
4. Equivalent TensorFlow implementation

## Something I Want to Say
1. What did we achieve? With the DNN class, we reached a training accuracy of 99.01% and a test accuracy of 98.77%. These are encouraging results, and the small gap suggests that the model generalizes well on this test set.
2. Why not use existing tools such as PyTorch or TensorFlow? First, because of CURIOSITY and ENTHUSIASM! When I first learned TensorFlow, I was curious about the logic behind it, so I wanted to understand how everything works. Second, I think it's always beneficial to know what's going on behind these tools. When we face problems or want to make changes in the future, knowing how things work and where we can make modifications can be very helpful.
3. Why write this notebook? I wrote this notebook to record what I came to understand and what I could improve, in the hope that these notes might also help others.
4. Finally, THANK YOU for reading! I'm still a passionate high school student, so there may be mistakes in my project. I would appreciate your valuable suggestions and feedback. THANK YOU!
