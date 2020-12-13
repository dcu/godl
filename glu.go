package tabnet

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
)

// GLUOpts are the supported options for GLU
type GLUOpts struct {
	VirtualBatchSize int
	OutputFeatures   int
	ActivationFn     ActivationFn
	FC               Layer
	WeightsInit      gorgonia.InitWFn
}

// GLU implements a Gated Linear Unit Block
func (nn *Model) GLU(opts GLUOpts) Layer {
	if opts.ActivationFn == nil {
		opts.ActivationFn = gorgonia.Sigmoid
	}

	return func(nodes ...*gorgonia.Node) (*gorgonia.Node, error) {
		x := nodes[0]

		log.Printf("%v", x.Shape())

		var (
			fc  *gorgonia.Node
			err error
		)

		if opts.FC == nil {
			opts.FC = nn.FC(FCOpts{
				OutputFeatures: opts.OutputFeatures * 2,
				WeightsInit:    opts.WeightsInit,
			})
		}

		fc, err = opts.FC(x)
		if err != nil {
			return nil, fmt.Errorf("[glu] applying FC(%v) failed: %w", x.Shape(), err)
		}

		log.Printf("fc shape: %v", fc.Shape())

		gbn, err := nn.GBN(GBNOpts{
			VirtualBatchSize: opts.VirtualBatchSize,
		})(fc)
		if err != nil {
			return nil, fmt.Errorf("[glu] applying GBN failed: %w", err)
		}

		// GLU
		firstHalf := gorgonia.Must(gorgonia.Slice(gbn, nil, gorgonia.S(0, opts.OutputFeatures)))
		secondHalf := gorgonia.Must(gorgonia.Slice(gbn, nil, gorgonia.S(opts.OutputFeatures, gbn.Shape()[1])))

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
