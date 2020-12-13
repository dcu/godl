package tabnet

import (
	"log"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestGBN(t *testing.T) {
	g := NewGraph()

	testCases := []struct {
		desc           string
		input          *Node
		vbs            int
		expectedShape  tensor.Shape
		expectedErr    string
		expectedOutput []float64
	}{
		{
			desc: "Example 1",
			input: NewTensor(g, tensor.Float64, 2, WithShape(1, 10), WithName("input"), WithValue(
				tensor.New(
					tensor.WithShape(1, 10),
					tensor.WithBacking([]float64{0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4}),
				),
			)),
			vbs:            5,
			expectedShape:  tensor.Shape{1, 10},
			expectedOutput: []float64{-1.4142100268524476, -0.7071050134262239, -3.140177066934696e-16, 0.7071050134262233, 1.4142100268524473, -1.4142100268524473, -0.7071050134262237, 0, 0.7071050134262237, 1.4142100268524473},
		},
	}

	tn := &Model{g: g}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)
			y, err := tn.GBN(GBNOpts{
				VirtualBatchSize: tcase.vbs,
			})(tcase.input)

			if tcase.expectedErr != "" {
				c.Error(err)

				c.Equal(tcase.expectedErr, err.Error())

				return
			}

			c.NoError(err)
			c.Equal(tcase.expectedShape, y.Shape())

			vm := NewTapeMachine(tn.g,
				gorgonia.WithLogger(log.New(os.Stderr, "[gorgonia]", log.LstdFlags)),
				gorgonia.BindDualValues(tn.learnables...),
				// gorgonia.WithValueFmt("%+v"),
				// gorgonia.WithWatchlist(),
			)
			c.NoError(vm.RunAll())

			c.Equal(tcase.expectedOutput, y.Value().Data().([]float64))
		})
	}
}
