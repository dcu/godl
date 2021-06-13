package tabnet

import (
	"math"

	"github.com/dcu/godl"
	"gorgonia.org/gorgonia"
)

// FeatureTransformerOpts contains options for feature transformer layer
type FeatureTransformerOpts struct {
	Shared            []godl.Layer
	VirtualBatchSize  int
	IndependentBlocks int
	InputDimension    int
	OutputDimension   int
	WithBias          bool
	Momentum          float32

	WeightsInit gorgonia.InitWFn
}

func (o *FeatureTransformerOpts) setDefaults() {
	if o.InputDimension == 0 {
		panic("input dimension can't be nil")
	}

	if o.OutputDimension == 0 {
		panic("output dimension can't be nil")
	}

	if o.Momentum == 0 {
		o.Momentum = 0.01
	}
}

// FeatureTransformer implements a feature transformer layer
func FeatureTransformer(nn *godl.Model, opts FeatureTransformerOpts) godl.Layer {
	lt := godl.AddLayer("FeatureTransformer")

	opts.setDefaults()

	shared := make([]godl.Layer, 0, len(opts.Shared))

	gluInput := opts.InputDimension
	gluOutput := opts.OutputDimension

	for _, fcLayer := range opts.Shared {
		weightsInit := opts.WeightsInit

		if weightsInit == nil {
			gain := math.Sqrt(float64(gluInput+gluOutput*2) / math.Sqrt(float64(gluInput)))
			weightsInit = gorgonia.GlorotN(gain)
		}

		shared = append(shared, godl.GLU(nn, godl.GLUOpts{
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

	independentBlocks := opts.IndependentBlocks
	gluOutput = opts.OutputDimension

	independent := make([]godl.Layer, 0, len(opts.Shared))
	for i := 0; i < independentBlocks; i++ {
		weightsInit := opts.WeightsInit

		if weightsInit == nil {
			gain := math.Sqrt(float64(gluInput+gluOutput*2) / math.Sqrt(float64(gluInput)))
			weightsInit = gorgonia.GlorotN(gain)
		}

		independent = append(independent, godl.GLU(nn, godl.GLUOpts{
			InputDimension:   gluInput,
			OutputDimension:  gluOutput,
			VirtualBatchSize: opts.VirtualBatchSize,
			WeightsInit:      weightsInit,
			WithBias:         opts.WithBias,
			Momentum:         opts.Momentum,
		}))
	}

	scale := gorgonia.NewConstant(float32(math.Sqrt(0.5)), gorgonia.WithName("ft.scale"))

	return func(nodes ...*gorgonia.Node) (godl.Result, error) {
		if err := nn.CheckArity(lt, nodes, 1); err != nil {
			return godl.Result{}, err
		}

		x := nodes[0]

		if len(shared) > 0 {
			result, err := shared[0](x)
			if err != nil {
				return godl.Result{}, err
			}

			x = result.Output

			for _, glu := range shared[1:] {
				result, err := glu(x)
				if err != nil {
					return godl.Result{}, godl.ErrorF(lt, "executing shared GLU layer with %v: %w", x.Shape(), err)
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
		}

		for _, layer := range independent {
			result, err := layer(x)
			if err != nil {
				return godl.Result{}, godl.ErrorF(lt, "executing independent GLU layer with %v: %w", x.Shape(), err)
			}

			x, err = gorgonia.Add(x, result.Output)
			if err != nil {
				return godl.Result{}, godl.ErrorF(lt, "add op: %w", err)
			}

			x, err = gorgonia.Mul(x, scale)
			if err != nil {
				return godl.Result{}, err
			}
		}

		return godl.Result{Output: x}, nil
	}
}
