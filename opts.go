package tabnet

import "gorgonia.org/gorgonia"

// ActivationFn represents an activation function
type ActivationFn func(*gorgonia.Node) (*gorgonia.Node, error)

// FCOpts contains optional parameter for a layer
type FCOpts struct {
	ActivationFn   ActivationFn
	Dropout        float64
	OutputFeatures int
}

// GBNOpts contains config options for the ghost batch normalization
type GBNOpts struct {
	Momentum         float64
	Epsilon          float64
	VirtualBatchSize int
}

func (o *GBNOpts) setDefaults() {
	if o.VirtualBatchSize == 0 {
		o.VirtualBatchSize = 128
	}

	if o.Momentum == 0.0 {
		o.Momentum = 0.01
	}

	if o.Epsilon == 0.0 {
		o.Epsilon = 1e-5
	}
}
