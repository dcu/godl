package tabnet

import (
	"math"

	"gorgonia.org/gorgonia"
)

// FeatureTransformerOpts contains options for feature transformer layer
type FeatureTransformerOpts struct {
	Shared            []Layer
	VirtualBatchSize  int
	IndependentBlocks int
	InputDimension    int
	OutputDimension   int
	Inferring         bool
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
}

// FeatureTransformer implements a feature transformer layer
func FeatureTransformer(nn *Model, opts FeatureTransformerOpts) Layer {
	lt := AddLayer("FeatureTransformer")

	opts.setDefaults()

	shared := make([]Layer, 0, len(opts.Shared))

	gluInput := opts.InputDimension
	gluOutput := opts.OutputDimension

	for _, fcLayer := range opts.Shared {
		weightsInit := opts.WeightsInit

		if weightsInit == nil {
			gain := math.Sqrt(float64(gluInput+gluOutput*2) / math.Sqrt(float64(gluInput)))
			weightsInit = gorgonia.GlorotN(gain)
		}

		shared = append(shared, GLU(nn, GLUOpts{
			InputDimension:   gluInput,
			OutputDimension:  gluOutput,
			VirtualBatchSize: opts.VirtualBatchSize,
			FC:               fcLayer,
			WeightsInit:      weightsInit,
			Inferring:        opts.Inferring,
			WithBias:         opts.WithBias,
			Momentum:         opts.Momentum,
		}))

		gluInput = gluOutput
	}

	independentBlocks := opts.IndependentBlocks
	gluOutput = opts.OutputDimension

	independent := make([]Layer, 0, len(opts.Shared))
	for i := 0; i < independentBlocks; i++ {
		weightsInit := opts.WeightsInit

		if weightsInit == nil {
			gain := math.Sqrt(float64(gluInput+gluOutput*2) / math.Sqrt(float64(gluInput)))
			weightsInit = gorgonia.GlorotN(gain)
		}

		independent = append(independent, GLU(nn, GLUOpts{
			InputDimension:   gluInput,
			OutputDimension:  gluOutput,
			VirtualBatchSize: opts.VirtualBatchSize,
			WeightsInit:      weightsInit,
			Inferring:        opts.Inferring,
			WithBias:         opts.WithBias,
			Momentum:         opts.Momentum,
		}))
	}

	scale := gorgonia.NewConstant(float32(math.Sqrt(0.5)), gorgonia.WithName("ft.scale"))

	return func(nodes ...*gorgonia.Node) (Result, error) {
		if err := nn.CheckArity(lt, nodes, 1); err != nil {
			return Result{}, err
		}

		x := nodes[0]

		if len(shared) > 0 {
			result, err := shared[0](x)
			if err != nil {
				return Result{}, err
			}

			x = result.Output

			for _, glu := range shared[1:] {
				result, err := glu(x)
				if err != nil {
					return Result{}, errorF(lt, "executing shared GLU layer with %v: %w", x.Shape(), err)
				}

				xShape := x.Shape()
				x, err = gorgonia.Add(x, result.Output)
				if err != nil {
					return Result{}, errorF(lt, "%v + %v: %w", xShape, result.Shape(), err)
				}

				x, err = gorgonia.Mul(x, scale)
				if err != nil {
					return Result{}, err
				}
			}
		}

		for _, layer := range independent {
			result, err := layer(x)
			if err != nil {
				return Result{}, errorF(lt, "executing independent GLU layer with %v: %w", x.Shape(), err)
			}

			x, err = gorgonia.Add(x, result.Output)
			if err != nil {
				return Result{}, errorF(lt, "add op: %w", err)
			}

			x, err = gorgonia.Mul(x, scale)
			if err != nil {
				return Result{}, err
			}
		}

		return Result{Output: x}, nil
	}
}
