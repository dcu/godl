package tabnet

import (
	"fmt"

	"gorgonia.org/gorgonia"
)

// AttentiveTransformer implements an attetion transformer layer
func (nn *TabNet) AttentiveTransformer(x *gorgonia.Node, prior *gorgonia.Node, fcOpts FCOpts, gbnOpts GBNOpts) (*gorgonia.Node, error) {
	fc, err := nn.FC(x, fcOpts)
	if err != nil {
		return nil, fmt.Errorf("fc(%v) failed failed: %w", x.Shape(), err)
	}

	bn, err := nn.GBN(fc, gbnOpts)
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
