package godl

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

	WeightsInit           gorgonia.InitWFn
	BiasInit              gorgonia.InitWFn
	WithBias              bool
	WeightsName, BiasName string
	FixedWeights          bool
}

func FC(nn *Model, opts FCOpts) Layer {
	lt := AddLayer("FC")

	MustBeGreatherThan(lt, "input dimension", opts.InputDimension, 0)
	MustBeGreatherThan(lt, "output dimension", opts.OutputDimension, 0)

	var (
		bias *gorgonia.Node
		w    = nn.AddWeights(lt, tensor.Shape{opts.InputDimension, opts.OutputDimension}, NewWeightsOpts{
			InitFN:     opts.WeightsInit,
			UniqueName: opts.WeightsName,
			Fixed:      opts.FixedWeights,
		})
	)

	if opts.WithBias {
		bias = nn.AddBias(lt, tensor.Shape{1, opts.OutputDimension}, NewWeightsOpts{
			InitFN:     opts.BiasInit,
			UniqueName: opts.BiasName,
			Fixed:      opts.FixedWeights,
		})
	}

	return func(inputs ...*gorgonia.Node) (Result, error) {
		err := nn.CheckArity(lt, inputs, 1)
		if err != nil {
			return Result{}, err
		}

		x := inputs[0]
		xShape := x.Shape()

		if x.Dims() > 2 {
			b, v := xShape[0], tensor.Shape(xShape[1:]).TotalSize()
			x = gorgonia.Must(gorgonia.Reshape(x, tensor.Shape{b, v}))
		}

		layer, err := gorgonia.Mul(x, w)
		if err != nil {
			return Result{}, ErrorF(lt, "error applying mul %v x %v: %w ", x.Shape(), w.Shape(), err)
		}

		if opts.WithBias {
			layer, err = gorgonia.BroadcastAdd(layer, bias, nil, []byte{0})
			if err != nil {
				return Result{}, ErrorF(lt, "error adding bias %w", err)
			}
		}

		if opts.Activation != nil {
			layer, err = opts.Activation(layer)
			if err != nil {
				return Result{}, ErrorF(lt, "error applying activation %w", err)
			}
		}

		if opts.Dropout > 0.0 {
			layer, err = gorgonia.Dropout(layer, opts.Dropout)
			if err != nil {
				return Result{}, ErrorF(lt, "error applying dropout %w", err)
			}
		}

		return Result{Output: layer}, nil
	}
}
