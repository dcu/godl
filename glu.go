package tabnet

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
func (nn *Model) GLU(opts GLUOpts) Layer {
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

	lt := incLayer("GLU")

	if opts.FC == nil {
		opts.FC = nn.FC(FCOpts{
			OutputDimension: opts.OutputDimension * 2,
			InputDimension:  opts.InputDimension,
			WeightsInit:     opts.WeightsInit,
			WithBias:        opts.WithBias,
		})
	}

	gbnLayer := nn.GBN(GBNOpts{
		VirtualBatchSize: opts.VirtualBatchSize,
		OutputDimension:  opts.OutputDimension * 2,
		Inferring:        opts.Inferring,
		Momentum:         opts.Momentum,
	})

	return func(nodes ...*gorgonia.Node) (*gorgonia.Node, *gorgonia.Node, error) {
		if err := nn.checkArity(lt, nodes, 1); err != nil {
			return nil, nil, err
		}

		x := nodes[0]

		var (
			fc  *gorgonia.Node
			err error
		)

		fc, _, err = opts.FC(x)
		if err != nil {
			return nil, nil, errorF(lt, "applying FC(%v) failed: %w", x.Shape(), err)
		}

		gbn, _, err := gbnLayer(fc)
		if err != nil {
			return nil, nil, errorF(lt, "applying GBN failed: %w", err)
		}

		// GLU
		firstHalf := gorgonia.Must(gorgonia.Slice(gbn, nil, gorgonia.S(0, opts.OutputDimension)))
		secondHalf := gorgonia.Must(gorgonia.Slice(gbn, nil, gorgonia.S(opts.OutputDimension, gbn.Shape()[1])))

		act, err := opts.ActivationFn(secondHalf)
		if err != nil {
			return nil, nil, errorF(lt, "%s: applying activation function failed: %w", err)
		}

		mul, err := gorgonia.HadamardProd(firstHalf, act)
		if err != nil {
			return nil, nil, errorF(lt, "%s: HadamardProd %d x %d: %w", firstHalf.Shape(), act.Shape(), err)
		}

		return mul, nil, nil
	}
}
