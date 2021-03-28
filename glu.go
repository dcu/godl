package godl

import (
	"gorgonia.org/gorgonia"
)

var gluCount uint64 = 0

// GLUOpts are the supported options for GLU
type GLUOpts struct {
	InputDimension   int
	OutputDimension  int
	VirtualBatchSize int
	ActivationFn     ActivationFn
	FC               Layer
	WeightsInit      gorgonia.InitWFn
	Inferring        bool
	WithBias         bool
	Momentum         float32
}

// GLU implements a Gated Linear Unit Block
func GLU(nn *Model, opts GLUOpts) Layer {
	if opts.ActivationFn == nil {
		opts.ActivationFn = gorgonia.Sigmoid
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

	lt := AddLayer("GLU")

	if opts.FC == nil {
		opts.FC = FC(nn, FCOpts{
			OutputDimension: opts.OutputDimension * 2,
			InputDimension:  opts.InputDimension,
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

		act, err := opts.ActivationFn(secondHalf)
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
