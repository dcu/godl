package tabnet

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// FCOpts contains optional parameter for a layer
type FCOpts struct {
	Activation      ActivationFn
	Dropout         float64
	OutputDimension int
	InputDimension  int

	WeightsInit gorgonia.InitWFn
	WithBias    bool
}

func FC(nn *Model, opts FCOpts) Layer {
	lt := incLayer("FC")

	mustBeGreaterThan(lt, "input dimension", opts.InputDimension, 0)
	mustBeGreaterThan(lt, "output dimension", opts.OutputDimension, 0)

	var (
		bias *gorgonia.Node
		w    = nn.addWeights(lt, tensor.Shape{opts.InputDimension, opts.OutputDimension}, opts.WeightsInit)
	)

	if opts.WithBias {
		bias = nn.addBias(lt, tensor.Shape{1, opts.OutputDimension}, opts.WeightsInit)
	}

	return func(inputs ...*gorgonia.Node) (*gorgonia.Node, *gorgonia.Node, error) {
		err := nn.checkArity(lt, inputs, 1)
		if err != nil {
			return nil, nil, err
		}

		x := inputs[0]
		xShape := x.Shape()

		if x.Dims() > 2 {
			b, v := xShape[0], tensor.Shape(xShape[1:]).TotalSize()
			x = gorgonia.Must(gorgonia.Reshape(x, tensor.Shape{b, v}))
		}

		layer, err := gorgonia.Mul(x, w)
		if err != nil {
			return nil, nil, errorF(lt, "error applying mul %v x %v: %w ", x.Shape(), w.Shape(), err)
		}

		if opts.WithBias {
			layer, err = gorgonia.BroadcastAdd(layer, bias, nil, []byte{0})
			if err != nil {
				return nil, nil, errorF(lt, "error adding bias %w", err)
			}
		}

		if opts.Activation != nil {
			layer, err = opts.Activation(layer)
			if err != nil {
				return nil, nil, errorF(lt, "error applying activation %w", err)
			}
		}

		if opts.Dropout > 0.0 {
			layer, err = gorgonia.Dropout(layer, opts.Dropout)
			if err != nil {
				return nil, nil, errorF(lt, "error applying dropout %w", err)
			}
		}

		return layer, nil, nil
	}
}
