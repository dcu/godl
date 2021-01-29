package tabnet

import (
	"gorgonia.org/gorgonia"
)

type MSELossOpts struct {
	Reduction string
}

// MSELoss defines the mean square root cost function
func MSELoss(output *gorgonia.Node, target *gorgonia.Node, opts MSELossOpts) *gorgonia.Node {
	reduction := gorgonia.Mean
	if opts.Reduction == "sum" {
		reduction = gorgonia.Sum
	}

	sub := gorgonia.Must(gorgonia.Sub(output, target))

	return gorgonia.Must(reduction(gorgonia.Must(gorgonia.Square(sub))))
}

type CrossEntropyLossOpt struct {
	Reduction string
}

// CrossEntropyLoss implements cross entropy loss function
func CrossEntropyLoss(output *gorgonia.Node, target *gorgonia.Node, opts CrossEntropyLossOpt) *gorgonia.Node {
	reduction := gorgonia.Mean
	if opts.Reduction == "sum" {
		reduction = gorgonia.Sum
	}

	output = gorgonia.Must(Softmax(output))
	cost := gorgonia.Must(gorgonia.HadamardProd(gorgonia.Must(gorgonia.Neg(gorgonia.Must(gorgonia.Log(output)))), target))

	return gorgonia.Must(reduction(cost))
}
