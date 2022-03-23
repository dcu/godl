package godl

import (
	"gorgonia.org/gorgonia"
)

func Sigmoid() Layer {
	return func(inputs ...*gorgonia.Node) (result Result, err error) {
		x := inputs[0]

		r, err := gorgonia.Sigmoid(x)
		if err != nil {
			return Result{}, err
		}

		return Result{Output: r}, nil
	}
}

func Tanh() Layer {
	return func(inputs ...*gorgonia.Node) (result Result, err error) {
		x := inputs[0]

		r, err := gorgonia.Tanh(x)
		if err != nil {
			return Result{}, err
		}

		return Result{Output: r}, nil
	}
}

func Rectify() Layer {
	return func(inputs ...*gorgonia.Node) (result Result, err error) {
		x := inputs[0]

		r, err := gorgonia.Rectify(x)
		if err != nil {
			return Result{}, err
		}

		return Result{Output: r}, nil
	}
}

func SparseMax(axis ...int) Layer {
	return func(inputs ...*gorgonia.Node) (result Result, err error) {
		x := inputs[0]

		r, err := gorgonia.Sparsemax(x, axis...)
		if err != nil {
			return Result{}, err
		}

		return Result{Output: r}, nil
	}
}

func SoftMax(axis ...int) Layer {
	return func(inputs ...*gorgonia.Node) (result Result, err error) {
		x := inputs[0]

		r, err := gorgonia.SoftMax(x, axis...)
		if err != nil {
			return Result{}, err
		}

		return Result{Output: r}, nil
	}
}
