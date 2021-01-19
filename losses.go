package tabnet

import "gorgonia.org/gorgonia"

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
