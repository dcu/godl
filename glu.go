package tabnet

import (
	"fmt"

	"gorgonia.org/gorgonia"
)

// GLUOpts are the supported options for GLU
type GLUOpts struct {
	VirtualBatchSize int
	OutputDimension   int
	ActivationFn     ActivationFn
	FC               Layer
	WeightsInit      gorgonia.InitWFn
	Inferring        bool
}

// GLU implements a Gated Linear Unit Block
func (nn *Model) GLU(opts GLUOpts) Layer {
	if opts.ActivationFn == nil {
		opts.ActivationFn = gorgonia.Sigmoid
	}

	gbnLayer := nn.GBN(GBNOpts{
		VirtualBatchSize: opts.VirtualBatchSize,
		WeightsInit:      opts.WeightsInit,
		Inferring:        opts.Inferring,
	})

	return func(nodes ...*gorgonia.Node) (*gorgonia.Node, error) {
		if err := nn.checkArity("GLU", nodes, 1); err != nil {
			return nil, err
		}

		x := nodes[0]

		var (
			fc  *gorgonia.Node
			err error
		)

		if opts.FC == nil {
			opts.FC = nn.FC(FCOpts{
				OutputDimension: opts.OutputDimension * 2,
				WeightsInit:    opts.WeightsInit,
				WithBias:       false,
			})
		}

		fc, err = opts.FC(x)
		if err != nil {
			return nil, fmt.Errorf("[glu] applying FC(%v) failed: %w", x.Shape(), err)
		}

		gbn, err := gbnLayer(fc)
		if err != nil {
			return nil, fmt.Errorf("[glu] applying GBN failed: %w", err)
		}

		// GLU
		firstHalf := gorgonia.Must(gorgonia.Slice(gbn, nil, gorgonia.S(0, opts.OutputDimension)))
		secondHalf := gorgonia.Must(gorgonia.Slice(gbn, nil, gorgonia.S(opts.OutputDimension, gbn.Shape()[1])))

		act, err := opts.ActivationFn(secondHalf)
		if err != nil {
			return nil, fmt.Errorf("[glu] applying activation function failed: %w", err)
		}

		mul, err := gorgonia.HadamardProd(firstHalf, act)
		if err != nil {
			return nil, fmt.Errorf("[glu] HadamardProd %d x %d: %w", firstHalf.Shape(), act.Shape(), err)
		}

		return mul, nil
	}
}
