package tabnet

import (
	"log"
	"testing"

	"github.com/dcu/godl"
	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestTabNetEmbeddings(t *testing.T) {
	testCases := []struct {
		desc              string
		input             tensor.Tensor
		vbs               int
		independentBlocks int
		sharedBlocks      int
		output            int
		steps             int
		gamma             float64
		prediction        int
		attentive         int
		expectedShape     tensor.Shape
		expectedErr       string
		expectedOutput    []float64
		expectedGrad      []float64
		expectedCost      float64
		expectedAcumLoss  float64
	}{
		{
			desc: "Example 1",
			input: tensor.New(
				tensor.WithShape(4, 4),
				tensor.WithBacking([]float64{0.4, 1.4, 2.4, 0, 4.4, 5.4, 6.4, 1, 8.4, 9.4, 10.4, 2, 12.4, 13.4, 14.4, 3}),
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
			expectedOutput:    []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882},
			expectedGrad:      []float64{0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332},
			expectedCost:      223.90071540814404,
			expectedAcumLoss:  -1.6094379119341007,
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			tn := godl.NewModel()
			tn.Training = true

			g := tn.ExprGraph()

			x := gorgonia.NewTensor(g, tensor.Float64, 2, gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithName("Input"), gorgonia.WithValue(tcase.input))

			a := gorgonia.NewTensor(g, tensor.Float64, tcase.input.Dims(), gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithInit(gorgonia.Ones()), gorgonia.WithName("AttentiveX"))
			priors := gorgonia.NewTensor(g, tensor.Float64, tcase.input.Dims(), gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithInit(gorgonia.Ones()), gorgonia.WithName("Priors"))

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
				CatIdxs:            []int{3},
				CatDims:            []int{4},
				CatEmbDim:          []int{2},
			})(x, a, priors)

			y := result.Output

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
				gorgonia.WithNaNWatch(),
				gorgonia.WithInfWatch(),
			)
			c.NoError(vm.RunAll())

			tn.PrintWatchables()
			// fmt.Printf("%v\n", g.String())

			c.Equal(tcase.expectedShape, y.Shape())

			log.Printf("y: %#v", y.Value().Data())
			c.InDeltaSlice(tcase.expectedOutput, y.Value().Data().([]float64), 1e-5)

			yGrad, err := y.Grad()
			c.NoError(err)

			c.Equal(tcase.expectedGrad, yGrad.Data())
			c.InDelta(tcase.expectedCost, cost.Value().Data(), 1e-5)
			c.Equal(tcase.expectedAcumLoss, result.Loss.Value().Data())

			weightsByName := map[string]*gorgonia.Node{}

			for _, n := range tn.Learnables() {
				weightsByName[n.Name()] = n

				wGrad, err := n.Grad()
				c.NoError(err)
				log.Printf("%s: %v", n.Name(), wGrad.Data().([]float64)[0:2])
			}

			// {
			// 	w := weightsByName["BatchNorm1d_31.81.scale.1.5"]
			// 	wGrad, err := w.Grad()
			// 	c.NoError(err)
			// 	c.Equal([]float64{0.0024, 0.0024, 0.0024, 0.0000, 0.0000}, wGrad.Data(), w.Name())
			// }

			optim := gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.02))
			err = optim.Step(gorgonia.NodesToValueGrads(tn.Learnables()))
			c.NoError(err)

			// {
			// 	w := weightsByName["BatchNorm1d_1.1.scale.1.5"]
			// 	log.Printf("weight updated: %v\n\n\n", w.Value())
			// 	c.Equal([]float64{0.9800, 0.9800, 0.9800, 1.0000, 1.0000}, w.Value().Data(), w.Name())
			// }

		})
	}
}
