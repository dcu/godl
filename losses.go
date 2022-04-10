package godl

import (
	"gorgonia.org/gorgonia"
)

var (
	reductionMap = map[Reduction]func(*gorgonia.Node, ...int) (*gorgonia.Node, error){
		ReductionNone: noopReduction,
		ReductionMean: gorgonia.Mean,
		ReductionSum:  gorgonia.Sum,
	}
)

type Reduction string

func (r Reduction) Func() func(*gorgonia.Node, ...int) (*gorgonia.Node, error) {
	fn, ok := reductionMap[r]
	if !ok {
		return gorgonia.Mean
	}

	return fn
}

const (
	ReductionNone Reduction = "none"
	ReductionSum  Reduction = "sum"
	ReductionMean Reduction = "mean"
)

type MSELossOpts struct {
	Reduction Reduction
}

type CostFn func(output Nodes, target *Node) *Node

// MSELoss defines the mean square root cost function
func MSELoss(opts MSELossOpts) CostFn {
	return func(output Nodes, target *gorgonia.Node) *gorgonia.Node {
		sub := gorgonia.Must(gorgonia.Sub(output[0], target))

		return gorgonia.Must(opts.Reduction.Func()(gorgonia.Must(gorgonia.Square(sub))))
	}
}

type CrossEntropyLossOpt struct {
	Reduction Reduction
}

// CrossEntropyLoss implements cross entropy loss function
func CrossEntropyLoss(opts CrossEntropyLossOpt) CostFn {
	return func(output Nodes, target *Node) *gorgonia.Node {
		cost := gorgonia.Must(gorgonia.HadamardProd(gorgonia.Must(gorgonia.Neg(gorgonia.Must(gorgonia.Log(output[0])))), target))

		return gorgonia.Must(opts.Reduction.Func()(cost))
	}
}

// CategoricalCrossEntropyLoss is softmax + cce
func CategoricalCrossEntropyLoss(opts CrossEntropyLossOpt) CostFn {
	return func(output Nodes, target *Node) *Node {
		cost := gorgonia.Must(gorgonia.SoftMax(output[0]))

		return CrossEntropyLoss(opts)(Nodes{cost}, target)
	}
}

func noopReduction(n *gorgonia.Node, along ...int) (*gorgonia.Node, error) {
	return n, nil
}
