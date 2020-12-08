package tabnet

import (
	"fmt"

	"gorgonia.org/gorgonia"
)

// GLU implements a Gated Linear Unit Block
func (nn *TabNet) GLU(x *gorgonia.Node, opts GLUOpts) (*gorgonia.Node, error) {
	fc, err := nn.FC(x, FCOpts{
		OutputFeatures: opts.OutputFeatures * 2,
	})
	if err != nil {
		return nil, fmt.Errorf("[glu] applying FC failed: %w", err)
	}

	gbn, err := nn.GBN(fc, GBNOpts{
		VirtualBatchSize: opts.VirtualBatchSize,
	})
	if err != nil {
		return nil, fmt.Errorf("[glu] applying GBN failed: %w", err)
	}

	firstHalf := gorgonia.Must(gorgonia.Slice(gbn, gorgonia.S(0, opts.OutputFeatures)))
	secondHalf := gorgonia.Must(gorgonia.Slice(gbn, gorgonia.S(opts.OutputFeatures, gbn.Shape()[1])))

	act, err := opts.ActivationFn(secondHalf)
	if err != nil {
		return nil, fmt.Errorf("[glu] applying activation function failed: %w", err)
	}

	return gorgonia.Mul(firstHalf, act)
}
