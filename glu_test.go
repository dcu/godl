package tabnet

import (
	"testing"

	"github.com/stretchr/testify/require"
	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestGLU(t *testing.T) {
	g := NewGraph()

	testCases := []struct {
		desc          string
		input         *Node
		vbs           int
		output        int
		expectedShape tensor.Shape
		expectedErr   string
	}{
		{
			desc:          "Example 1",
			input:         NewMatrix(g, tensor.Float64, WithShape(100, 10), WithName("input"), WithInit(GlorotN(1.0))),
			vbs:           10,
			output:        10,
			expectedShape: tensor.Shape{10, 20},
		},
	}

	tn := &Model{g: g}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)
			x, err := tn.GLU(GLUOpts{
				VirtualBatchSize: tcase.vbs,
				OutputFeatures:   tcase.output,
			})(tcase.input)

			if tcase.expectedErr != "" {
				c.Error(err)

				c.Equal(tcase.expectedErr, err.Error())

				return
			}

			c.NoError(err)
			c.Equal(tcase.expectedShape, x.Shape())
		})
	}
}
