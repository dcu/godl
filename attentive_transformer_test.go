package tabnet

import (
	"testing"

	"github.com/stretchr/testify/require"
	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestAttentiveTransformer(t *testing.T) {
	g := NewGraph()

	testCases := []struct {
		desc           string
		input          *Node
		vbs            int
		output         int
		expectedShape  tensor.Shape
		expectedErr    string
		expectedOutput []float64
	}{
		{
			desc: "Example 1",
			input: NewTensor(g, tensor.Float64, 2, WithShape(5, 2), WithName("input"), WithValue(
				tensor.New(
					tensor.WithShape(5, 2),
					tensor.WithBacking([]float64{0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4}),
				),
			)),
			vbs:            2,
			output:         2,
			expectedShape:  tensor.Shape{5, 2},
			expectedOutput: []float64{0.3999845913169984, 0, 0.3999990275527686, 0, 0.39999996894071055, 0, 0.40000021751596193, 0, 0.4000003178664261, 0},
		},
	}

	tn := &Model{g: g}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			priors := NewTensor(g, Float64, tcase.input.Dims(), WithShape(tcase.input.Shape()...), WithInit(Ones()))
			x, err := tn.AttentiveTransformer(AttentiveTransformerOpts{
				VirtualBatchSize: tcase.vbs,
				OutputFeatures:   tcase.output,
				WeightsInit:      initDummyWeights,
			})(tcase.input, priors)

			if tcase.expectedErr != "" {
				c.Error(err)

				c.Equal(tcase.expectedErr, err.Error())

				return
			} else {
				c.NoError(err)
			}

			vm := NewTapeMachine(g)
			c.NoError(vm.RunAll())

			c.Equal(tcase.expectedShape, x.Shape())
			c.Equal(tcase.expectedOutput, x.Value().Data().([]float64))
		})
	}
}
