package tabnet

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestDecisionStep(t *testing.T) {
	g := NewGraph()

	testCases := []struct {
		desc              string
		input             *Node
		vbs               int
		independentBlocks int
		output            int
		expectedShape     tensor.Shape
		expectedErr       string
		expectedOutput    []float64
		expectedPriors    []float64
	}{
		{
			desc: "Example 1",
			input: NewTensor(g, tensor.Float64, 2, WithShape(2, 8), WithName("input"), WithValue(
				tensor.New(
					tensor.WithShape(2, 8),
					tensor.WithBacking([]float64{0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4}),
				),
			)),
			vbs:               2,
			output:            8,
			independentBlocks: 3,
			expectedShape:     tensor.Shape{2, 8},
			expectedOutput:    []float64{1.771626481833988, 0.33778769572757766, -0.012137622720752451, -0.5147066700602438, 1.016029500967262, 0.096918574054974, -0.2949884387331374, -1.5701877239725464, 1.771626481833988, 0.33778769572757766, -0.012137622720752451, -0.5147066700602438, 1.016029500967262, 0.096918574054974, -0.2949884387331374, -1.5701877239725464},
			expectedPriors:    []float64{0.37323545866602437, 0.7071067811865476, 0.7071067811865476, 0.7052770671980956, -0.06082574429874554, 0.7071067811865476, 0.7071067811865476, 0.7071067811865476, 0.37323545866602437, 0.7071067811865476, 0.7071067811865476, 0.7052770671980956, -0.06082574429874554, 0.7071067811865476, 0.7071067811865476, 0.7071067811865476},
		},
	}

	tn := &Model{g: g}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			a := NewTensor(g, Float64, tcase.input.Dims(), WithShape(tcase.input.Shape()...), WithInit(Ones()))
			priors := NewTensor(g, Float64, tcase.input.Dims(), WithShape(tcase.input.Shape()...), WithInit(Ones()))
			step := tn.DecisionStep(DecisionStepOpts{
				VirtualBatchSize:   tcase.vbs,
				Shared:             nil,
				IndependentBlocks:  tcase.independentBlocks,
				PredictionLayerDim: 0,
				AttentionLayerDim:  8,
				WeightsInit:        initDummyWeights,
				OutputDimension:    tcase.output,
			})

			mask, _, err := step.AttentiveTransformer(a, priors)
			c.NoError(err)

			// Update prior
			{
				gamma := gorgonia.NewScalar(tn.g, tensor.Float64, gorgonia.WithValue(1.2))
				priors, err = gorgonia.HadamardProd(priors, gorgonia.Must(gorgonia.Sub(gamma, mask)))
				c.NoError(err)
			}

			ds, _, err := step.FeatureTransformer(tcase.input, mask)
			c.NoError(err)

			if tcase.expectedErr != "" {
				c.Error(err)

				c.Equal(tcase.expectedErr, err.Error())

				return
			} else {
				c.NoError(err)
			}

			vm := NewTapeMachine(g,
				gorgonia.WithWatchlist(watchables...),
				gorgonia.WithLogger(testLogger),
				gorgonia.WithValueFmt("%v"),
				gorgonia.TraceExec(),
			)
			c.NoError(vm.RunAll())

			c.Equal(tcase.expectedShape, ds.Shape())
			c.Equal(tcase.expectedOutput, ds.Value().Data().([]float64))

			c.Equal(tcase.expectedShape, priors.Shape())
			c.Equal(tcase.expectedPriors, priors.Value().Data().([]float64))
		})
	}
}
