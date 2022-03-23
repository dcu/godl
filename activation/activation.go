package activation

import "gorgonia.org/gorgonia"

// Function represents an activation function
type Function func(*gorgonia.Node) (*gorgonia.Node, error)

func Sigmoid(x *gorgonia.Node) (*gorgonia.Node, error) {
	return gorgonia.Sigmoid(x)
}

func Tanh(x *gorgonia.Node) (*gorgonia.Node, error) {
	return gorgonia.Tanh(x)
}

func Rectify(x *gorgonia.Node) (*gorgonia.Node, error) {
	return gorgonia.Rectify(x)
}

func SoftMax(x *gorgonia.Node) (*gorgonia.Node, error) {
	return gorgonia.SoftMax(x)
}

func SparseMax(x *gorgonia.Node) (*gorgonia.Node, error) {
	return gorgonia.Sparsemax(x)
}
