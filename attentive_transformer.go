package tabnet

import (
	"fmt"

	"gorgonia.org/gorgonia"
)

type AttentiveTransformerOpts struct {
	InputDimension                   int
	OutputDimension                  int
	Momentum                         float64
	Epsilon                          float64
	VirtualBatchSize                 int
	Inferring                        bool
	Activation                       ActivationFn
	WithBias                         bool
	WeightsInit, ScaleInit, BiasInit gorgonia.InitWFn
}

func (o *AttentiveTransformerOpts) setDefaults() {
	if o.Activation == nil {
		o.Activation = sparsemax
	}
}

// AttentiveTransformer implements an attetion transformer layer
func (nn *Model) AttentiveTransformer(opts AttentiveTransformerOpts) Layer {
	opts.setDefaults()

	fcLayer := nn.FC(FCOpts{
		InputDimension:  opts.InputDimension,
		OutputDimension: opts.OutputDimension,
		WeightsInit:     opts.WeightsInit,
		WithBias:        opts.WithBias,
	})

	gbnLayer := nn.GBN(GBNOpts{
		Momentum:         opts.Momentum,
		Epsilon:          opts.Epsilon,
		VirtualBatchSize: opts.VirtualBatchSize,
		OutputDimension:        opts.OutputDimension,
		Inferring:        opts.Inferring,
		ScaleInit:        opts.ScaleInit,
		BiasInit:         opts.BiasInit,
		WeightsInit:      opts.WeightsInit,
	})

	return func(nodes ...*gorgonia.Node) (*gorgonia.Node, *gorgonia.Node, error) {
		if err := nn.checkArity("AttentiveTransformer", nodes, 2); err != nil {
			return nil, nil, err
		}

		x := nodes[0]
		prior := nodes[1]

		fc, _, err := fcLayer(x)
		if err != nil {
			return nil, nil, fmt.Errorf("AttentiveTransformer: fc%v failed failed: %w", x.Shape(), err)
		}

		bn, _, err := gbnLayer(fc)
		if err != nil {
			return nil, nil, fmt.Errorf("AttentiveTransformer: gbn%v failed: %w", fc.Shape(), err)
		}

		mul, err := gorgonia.HadamardProd(bn, prior)
		if err != nil {
			return nil, nil, fmt.Errorf("AttentiveTransformer: mul(%v, %v) failed: %w", bn.Shape(), prior.Shape(), err)
		}

		sm, err := opts.Activation(mul)
		if err != nil {
			return nil, nil, fmt.Errorf("AttentiveTransformer: sparsemax(%v) failed: %w", mul.Shape(), err)
		}

		return sm, nil, nil
	}
}
