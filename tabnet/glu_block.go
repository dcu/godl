package tabnet

import (
	"math"

	"github.com/dcu/godl"
	"gorgonia.org/gorgonia"
)

type GLUBlockOpts struct {
	InputDimension   int
	OutputDimension  int
	Shared           []godl.Layer
	VirtualBatchSize int

	Size int

	WithBias    bool
	Momentum    float32
	WeightsInit gorgonia.InitWFn
}

func GLUBlock(nn *godl.Model, opts GLUBlockOpts) godl.Layer {
	lt := godl.AddLayer("GLUBlock")

	gluLayers := make([]godl.Layer, 0, opts.Size)
	gluInput := opts.InputDimension
	if len(opts.Shared) == 0 { // for independent layers
		gluInput = opts.OutputDimension
	}

	gluOutput := opts.OutputDimension
	weightsInit := opts.WeightsInit

	if weightsInit == nil {
		gain := math.Sqrt(float64(gluInput+gluOutput) / math.Sqrt(float64(gluInput)))
		weightsInit = gorgonia.GlorotN(gain)
	}

	for i := 0; i < opts.Size; i++ {
		var fcLayer godl.Layer
		if len(opts.Shared) > 0 {
			fcLayer = opts.Shared[i]
		}

		gluLayers = append(gluLayers, godl.GLU(nn, godl.GLUOpts{
			InputDimension:   gluInput,
			OutputDimension:  gluOutput,
			VirtualBatchSize: opts.VirtualBatchSize,
			FC:               fcLayer,
			WeightsInit:      weightsInit,
			WithBias:         opts.WithBias,
			Momentum:         opts.Momentum,
		}))

		gluInput = gluOutput
	}

	scale := gorgonia.NewConstant(float32(math.Sqrt(0.5)), gorgonia.WithName("ft.scale"))

	return func(nodes ...*gorgonia.Node) (godl.Result, error) {
		if err := nn.CheckArity(lt, nodes, 1); err != nil {
			return godl.Result{}, err
		}

		x := nodes[0]
		startAt := 0

		if len(opts.Shared) > 0 {
			result, err := gluLayers[0](x)
			if err != nil {
				return godl.Result{}, err
			}

			x = result.Output
			startAt = 1
		}

		for _, glu := range gluLayers[startAt:] {
			result, err := glu(x)
			if err != nil {
				return godl.Result{}, godl.ErrorF(lt, "executing GLU layer with %v: %w", x.Shape(), err)
			}

			xShape := x.Shape()
			x, err = gorgonia.Add(x, result.Output)
			if err != nil {
				return godl.Result{}, godl.ErrorF(lt, "%v + %v: %w", xShape, result.Shape(), err)
			}

			x, err = gorgonia.Mul(x, scale)
			if err != nil {
				return godl.Result{}, err
			}
		}

		return godl.Result{Output: x}, nil
	}
}
