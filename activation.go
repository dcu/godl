package tabnet

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// ActivationFn represents an activation function
type ActivationFn func(*gorgonia.Node) (*gorgonia.Node, error)

func Sigmoid(x *gorgonia.Node) (*gorgonia.Node, error) {
	return gorgonia.Sigmoid(x)
}

func Rectify(x *gorgonia.Node) (*gorgonia.Node, error) {
	return gorgonia.Rectify(x)
}

// this is wrap because gorgonia.Sparsemax doesn't match the ActivationFn signature
func Sparsemax(x *gorgonia.Node) (*gorgonia.Node, error) {
	return gorgonia.Sparsemax(x)
}

// TODO: remove once gorgonia.SoftMax is fixed, currently it's broken on master
func Softmax(a *gorgonia.Node) (retVal *gorgonia.Node, err error) {
	aShape := a.Shape()
	axis := aShape.Dims() - 1 // default: last dim
	if a.IsColVec() || (a.IsVector() && !a.IsRowVec()) {
		axis = 0
	}

	var exp, sum *gorgonia.Node
	if exp, err = gorgonia.Exp(a); err != nil {
		return nil, err
	}

	if sum, err = gorgonia.Sum(exp, axis); err != nil {
		return nil, err
	}

	if sum.IsScalar() {
		return gorgonia.HadamardDiv(exp, sum)
	}

	ss := sum.Shape()
	diff := exp.Shape().Dims() - ss.Dims()

	if diff > 0 {
		newShape := tensor.Shape(tensor.BorrowInts(ss.Dims() + diff))
		copy(newShape, ss)
		copy(newShape[axis+1:], newShape[axis:])
		newShape[axis] = 1

		if sum, err = gorgonia.Reshape(sum, newShape); err != nil {
			return nil, err
		}
	}

	return gorgonia.BroadcastHadamardDiv(exp, sum, nil, []byte{byte(axis)})
}

var (
	_ ActivationFn = Sparsemax
	_ ActivationFn = Softmax
)
