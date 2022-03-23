package godl

import (
	"github.com/dcu/godl/activation"
	"gorgonia.org/gorgonia"
)

var gluCount uint64 = 0

// GLUOpts are the supported options for GLU
type GLUOpts struct {
	InputDimension   int
	OutputDimension  int
	VirtualBatchSize int
	Activation       activation.Function
	FC               Layer
	WeightsInit      gorgonia.InitWFn
	WithBias         bool
	Momentum         float64
}

func (opts *GLUOpts) setDefaults() {
	if opts.Momentum == 0 {
		opts.Momentum = 0.02
	}

	if opts.Activation == nil {
		opts.Activation = gorgonia.Sigmoid
	}

	if opts.InputDimension == 0 {
		panic("input dimension must be set")
	}

	if opts.OutputDimension == 0 {
		panic("output dimension must be set")
	}

	if opts.VirtualBatchSize == 0 {
		panic("virtual batch size must be set")
	}
}

// GLU implements a Gated Linear Unit Block
func GLU(nn *Model, opts GLUOpts) Layer {
	opts.setDefaults()

	lt := AddLayer("GLU")

	if opts.FC == nil {
		opts.FC = FC(nn, FCOpts{
			InputDimension:  opts.InputDimension,
			OutputDimension: opts.OutputDimension * 2,
			WeightsInit:     opts.WeightsInit,
			WithBias:        opts.WithBias,
		})
	}

	gbnLayer := GBN(nn, GBNOpts{
		VirtualBatchSize: opts.VirtualBatchSize,
		OutputDimension:  opts.OutputDimension * 2,
		Momentum:         opts.Momentum,
	})

	return func(nodes ...*gorgonia.Node) (Result, error) {
		if err := nn.CheckArity(lt, nodes, 1); err != nil {
			return Result{}, err
		}

		x := nodes[0]

		fcResult, err := opts.FC(x)
		if err != nil {
			return Result{}, ErrorF(lt, "applying FC(%v) failed: %w", x.Shape(), err)
		}

		gnbResult, err := gbnLayer(fcResult.Output)
		if err != nil {
			return Result{}, ErrorF(lt, "applying GBN failed: %w", err)
		}

		// GLU
		firstHalf := gorgonia.Must(gorgonia.Slice(gnbResult.Output, nil, gorgonia.S(0, opts.OutputDimension)))
		secondHalf := gorgonia.Must(gorgonia.Slice(gnbResult.Output, nil, gorgonia.S(opts.OutputDimension, gnbResult.Output.Shape()[1])))

		act, err := opts.Activation(secondHalf)
		if err != nil {
			return Result{}, ErrorF(lt, "%s: applying activation function failed: %w", err)
		}

		mul, err := gorgonia.HadamardProd(firstHalf, act)
		if err != nil {
			return Result{}, ErrorF(lt, "%s: HadamardProd %d x %d: %w", firstHalf.Shape(), act.Shape(), err)
		}

		return Result{Output: mul}, nil
	}
}
