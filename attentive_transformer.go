package tabnet

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
)

type AttentiveTransformerOpts struct {
	OutputFeatures                   int
	Momentum                         float64
	Epsilon                          float64
	VirtualBatchSize                 int
	Inferring                        bool
	WeightsInit, ScaleInit, BiasInit gorgonia.InitWFn
}

// AttentiveTransformer implements an attetion transformer layer
func (nn *Model) AttentiveTransformer(opts AttentiveTransformerOpts) Layer {
	fcLayer := nn.FC(FCOpts{
		OutputFeatures: opts.OutputFeatures,
		WeightsInit:    opts.WeightsInit,
	})
	gbnLayer := nn.GBN(GBNOpts{
		Momentum:         opts.Momentum,
		Epsilon:          opts.Epsilon,
		VirtualBatchSize: opts.VirtualBatchSize,
		Inferring:        opts.Inferring,
		ScaleInit:        opts.ScaleInit,
		BiasInit:         opts.BiasInit,
		WeightsInit:      opts.WeightsInit,
	})

	return func(nodes ...*gorgonia.Node) (*gorgonia.Node, error) {
		if err := nn.checkArity("AttentiveTransformer", nodes, 2); err != nil {
			return nil, err
		}

		x := nodes[0]
		xShape := x.Shape()
		prior := nodes[1]

		fc, err := fcLayer(x)
		if err != nil {
			return nil, fmt.Errorf("fc(%v) failed failed: %w", x.Shape(), err)
		}

		log.Printf("fc(%v) -> %v", x.Shape(), fc.Shape())

		bn, err := gbnLayer(fc)
		if err != nil {
			return nil, fmt.Errorf("gbn(%v) failed: %w", fc.Shape(), err)
		}

		log.Printf("bn(%v) -> %v", fc.Shape(), bn.Shape())

		priorR, err := gorgonia.Reshape(prior, bn.Shape())
		if err != nil {
			return nil, fmt.Errorf("reshape prior %v to %v failed: %w", prior.Shape(), bn.Shape(), err)
		}

		mul, err := gorgonia.HadamardProd(bn, priorR)
		if err != nil {
			return nil, fmt.Errorf("mul(%v, %v) failed: %w", bn.Shape(), priorR.Shape(), err)
		}

		log.Printf("spm(%v)", mul.Shape())
		sm, err := gorgonia.Sparsemax(mul)
		if err != nil {
			return nil, fmt.Errorf("sparsemax(%v) failed: %w", mul.Shape(), err)
		}

		return gorgonia.Reshape(sm, xShape)
	}
}
