package tabnet

import (
	"testing"

	"github.com/dcu/godl"
	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestDecisionStep(t *testing.T) {
	testCases := []struct {
		desc                string
		input               tensor.Tensor
		vbs                 int
		independentBlocks   int
		predictionLayerSize int
		attentiveLayerSize  int
		output              int
		expectedShape       tensor.Shape
		expectedErr         string
		expectedOutput      []float32
		expectedPriors      []float32
		expectedGrad        []float32
		expectedCost        float32
	}{
		{
			desc: "Example 1",
			input: tensor.New(
				tensor.WithShape(4, 7),
				tensor.WithBacking([]float32{0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 0.4, 1.4}),
			),
			vbs:                 2,
			independentBlocks:   3,
			predictionLayerSize: 0,
			attentiveLayerSize:  7,
			expectedShape:       tensor.Shape{4, 7},
			expectedOutput:      []float32{-0.399498, -0.34899038, -0.29848272, -0.24797511, -0.19746749, -0.14695986, -0.09645221, 1.5143689, 1.1608156, 1.2113231, 1.2618308, 1.3123384, 1.3628461, 1.4133536, 1.4641757, 1.5146835, 1.5651909, 1.6156987, 1.1611301, 1.2116376, 1.2621453, -0.24799965, -0.19749203, -0.1469844, -0.09647678, -0.045969147, -0.39952257, -0.34901494},
			expectedPriors:      []float32{1.0571429, 1.0571429, 1.0571429, 1.0571429, 1.0571429, 1.0571429, 1.0571429, 1.0571429, 1.0571429, 1.0571429, 1.0571429, 1.0571429, 1.0571429, 1.0571429, 1.0571429, 1.0571429, 1.0571429, 1.0571429, 1.0571429, 1.0571429, 1.0571429, 1.0571429, 1.0571429, 1.0571429, 1.0571429, 1.0571429, 1.0571429, 1.0571429},
			expectedGrad:        []float32{0.035714287, 0.035714287, 0.035714287, 0.035714287, 0.035714287, 0.035714287, 0.035714287, 0.035714287, 0.035714287, 0.035714287, 0.035714287, 0.035714287, 0.035714287, 0.035714287, 0.035714287, 0.035714287, 0.035714287, 0.035714287, 0.035714287, 0.035714287, 0.035714287, 0.035714287, 0.035714287, 0.035714287, 0.035714287, 0.035714287, 0.035714287, 0.035714287},
			expectedCost:        0.5647233,
		},
		{
			desc: "Example 2",
			input: tensor.New(
				tensor.WithShape(2, 8),
				tensor.WithBacking([]float32{0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4}),
			),
			vbs:                 2,
			independentBlocks:   3,
			predictionLayerSize: 0,
			attentiveLayerSize:  8,
			expectedShape:       tensor.Shape{2, 8},
			expectedOutput:      []float32{0.01767767, 0.06187184, 0.10606602, 0.1502602, 0.19445437, 0.23864852, 0.28284273, 0.3270369, 0.01767767, 0.06187184, 0.10606602, 0.1502602, 0.19445437, 0.23864852, 0.28284273, 0.3270369},
			expectedPriors:      []float32{1.075, 1.075, 1.075, 1.075, 1.075, 1.075, 1.075, 1.075, 1.075, 1.075, 1.075, 1.075, 1.075, 1.075, 1.075, 1.075},
			expectedGrad:        []float32{0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625},
			expectedCost:        0.17235726,
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			tn := godl.NewModel()
			tn.Training = true

			g := tn.ExprGraph()

			c := require.New(t)

			input := gorgonia.NewTensor(g, tensor.Float32, tcase.input.Dims(), gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithName("input"), gorgonia.WithValue(tcase.input))

			a := gorgonia.NewTensor(g, gorgonia.Float32, input.Dims(), gorgonia.WithShape(input.Shape()...), gorgonia.WithInit(gorgonia.Ones()))
			priors := gorgonia.NewTensor(g, gorgonia.Float32, input.Dims(), gorgonia.WithShape(input.Shape()...), gorgonia.WithInit(gorgonia.Ones()))
			step := NewDecisionStep(tn, DecisionStepOpts{
				VirtualBatchSize:   tcase.vbs,
				Shared:             nil,
				IndependentBlocks:  tcase.independentBlocks,
				PredictionLayerDim: tcase.predictionLayerSize,
				AttentionLayerDim:  tcase.attentiveLayerSize,
				WeightsInit:        initDummyWeights,
				InputDimension:     a.Shape()[0],
				OutputDimension:    a.Shape()[1],
			})

			mask, err := step.AttentiveTransformer(a, priors)
			c.NoError(err)

			// Update prior
			{
				gamma := gorgonia.NewScalar(tn.ExprGraph(), tensor.Float32, gorgonia.WithValue(float32(1.2)))
				priors, err = gorgonia.HadamardProd(priors, gorgonia.Must(gorgonia.Sub(gamma, mask.Output)))
				c.NoError(err)
			}

			newInput := gorgonia.Must(gorgonia.HadamardProd(input, mask.Output))
			y, err := step.FeatureTransformer(newInput)
			c.NoError(err)

			if tcase.expectedErr != "" {
				c.Error(err)

				c.Equal(tcase.expectedErr, err.Error())

				return
			} else {
				c.NoError(err)
			}

			cost := gorgonia.Must(gorgonia.Mean(y.Output))
			_, err = gorgonia.Grad(cost, input)
			c.NoError(err)

			vm := gorgonia.NewTapeMachine(g,
				gorgonia.WithLogger(testLogger),
				gorgonia.WithValueFmt("%v"),
				gorgonia.TraceExec(),
			)
			c.NoError(vm.RunAll())

			tn.PrintWatchables()

			c.Equal(tcase.expectedShape, y.Shape())
			c.Equal(tcase.expectedOutput, y.Value().Data().([]float32))

			c.Equal(tcase.expectedShape, priors.Shape())
			c.Equal(tcase.expectedPriors, priors.Value().Data().([]float32))

			yGrad, err := y.Output.Grad()
			c.NoError(err)

			c.Equal(tcase.expectedGrad, yGrad.Data())
			c.Equal(tcase.expectedCost, cost.Value().Data())
		})
	}
}
