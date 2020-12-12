package tabnet

import (
	"fmt"

	"gorgonia.org/gorgonia"
)

type AttentiveTransformerOpts struct {
	OutputFeatures      int
	Momentum            float64
	Epsilon             float64
	VirtualBatchSize    int
	Inferring           bool
	ScaleInit, BiasInit gorgonia.InitWFn
}

// AttentiveTransformer implements an attetion transformer layer
func (nn *Model) AttentiveTransformer(opts AttentiveTransformerOpts) Layer {
	fcLayer := nn.FC(FCOpts{
		OutputFeatures: opts.OutputFeatures,
	})
	gbnLayer := nn.GBN(GBNOpts{
		Momentum:         opts.Momentum,
		Epsilon:          opts.Epsilon,
		VirtualBatchSize: opts.VirtualBatchSize,
		Inferring:        opts.Inferring,
		ScaleInit:        opts.ScaleInit,
		BiasInit:         opts.BiasInit,
	})

	return func(nodes ...*gorgonia.Node) (*gorgonia.Node, error) {
		x := nodes[0]
		prior := nodes[1]

		fc, err := fcLayer(x)
		if err != nil {
			return nil, fmt.Errorf("fc(%v) failed failed: %w", x.Shape(), err)
		}

		bn, err := gbnLayer(fc)
		if err != nil {
			return nil, fmt.Errorf("gbn(%v) failed: %w", x.Shape(), err)
		}

		mul, err := gorgonia.Mul(bn, prior)
		if err != nil {
			return nil, fmt.Errorf("mul(%v, %v) failed: %w", x.Shape(), prior.Shape(), err)
		}

		sm, err := gorgonia.Sparsemax(mul)
		if err != nil {
			return nil, fmt.Errorf("sparsemax(%v) failed: %w", x.Shape(), err)
		}

		return sm, nil
	}
}
