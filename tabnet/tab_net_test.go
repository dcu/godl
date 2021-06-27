package tabnet

import (
	"testing"

	"github.com/dcu/godl"
	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestTabNet(t *testing.T) {
	testCases := []struct {
		desc              string
		input             tensor.Tensor
		vbs               int
		independentBlocks int
		sharedBlocks      int
		output            int
		steps             int
		gamma             float32
		prediction        int
		attentive         int
		expectedShape     tensor.Shape
		expectedErr       string
		expectedOutput    []float32
		expectedGrad      []float32
		expectedCost      float32
		expectedAcumLoss  float32
	}{
		{
			desc: "Example 1",
			input: tensor.New(
				tensor.WithShape(4, 4),
				tensor.WithBacking([]float32{0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4}),
			),
			vbs:               2,
			output:            12,
			independentBlocks: 2,
			sharedBlocks:      2,
			steps:             5,
			gamma:             1.2,
			prediction:        64,
			attentive:         64,
			expectedShape:     tensor.Shape{4, 12},
			expectedOutput:    []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062},
			expectedGrad:      []float32{0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334, 0.020833334},
			expectedCost:      223.90314,
			expectedAcumLoss:  -1.3862944,
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			tn := godl.NewModel()
			tn.Training = true

			g := tn.ExprGraph()

			y := gorgonia.NewTensor(g, tensor.Float32, 2, gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithName("Input"), gorgonia.WithValue(tcase.input))

			a := gorgonia.NewTensor(g, tensor.Float32, tcase.input.Dims(), gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithInit(gorgonia.Ones()), gorgonia.WithName("AttentiveX"))
			priors := gorgonia.NewTensor(g, tensor.Float32, tcase.input.Dims(), gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithInit(gorgonia.Ones()), gorgonia.WithName("Priors"))

			result, err := TabNet(tn, TabNetOpts{
				VirtualBatchSize:   tcase.vbs,
				IndependentBlocks:  tcase.independentBlocks,
				PredictionLayerDim: tcase.prediction,
				AttentionLayerDim:  tcase.attentive,
				OutputSize:         tcase.output,
				SharedBlocks:       tcase.sharedBlocks,
				DecisionSteps:      tcase.steps,
				Gamma:              tcase.gamma,
				InputSize:          a.Shape()[0],
				BatchSize:          a.Shape()[0],
				WeightsInit:        initDummyWeights,
				ScaleInit:          gorgonia.Ones(),
				BiasInit:           gorgonia.Zeroes(),
				Epsilon:            1e-10,
			})(y, a, priors)

			y = result.Output

			if tcase.expectedErr != "" {
				c.Error(err)

				c.Equal(tcase.expectedErr, err.Error())

				return
			} else {
				c.NoError(err)
			}

			cost := gorgonia.Must(gorgonia.Mean(y))
			_, err = gorgonia.Grad(cost, tn.Learnables()...)
			c.NoError(err)

			vm := gorgonia.NewTapeMachine(g,
				gorgonia.BindDualValues(tn.Learnables()...),
				gorgonia.WithLogger(testLogger),
				gorgonia.WithValueFmt("%+v"),
				gorgonia.WithWatchlist(),
			)
			c.NoError(vm.RunAll())

			tn.PrintWatchables()
			// fmt.Printf("%v\n", g.String())

			c.Equal(tcase.expectedShape, y.Shape())
			c.Equal(tcase.expectedOutput, y.Value().Data().([]float32))

			yGrad, err := y.Grad()
			c.NoError(err)

			c.Equal(tcase.expectedGrad, yGrad.Data())
			c.Equal(tcase.expectedCost, cost.Value().Data())
			c.Equal(tcase.expectedAcumLoss, result.Loss.Value().Data())
		})
	}
}
