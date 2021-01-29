package tabnet

import "gorgonia.org/gorgonia"

// Sequential runs the given layers one after the other
func Sequential(m *Model, layers ...Layer) Layer {
	return func(inputs ...*gorgonia.Node) (Result, error) {
		err := m.CheckArity("Sequential", inputs, 1)
		if err != nil {
			return Result{}, err
		}

		x := inputs[0]
		losses := make([]*gorgonia.Node, 0, len(layers))

		for _, layer := range layers {
			result, err := layer(x)
			if err != nil {
				return Result{}, err
			}

			x = result.Output

			if result.Loss != nil {
				losses = append(losses, result.Loss)
			}
		}

		if len(losses) == 0 {
			return Result{Output: x}, nil
		}

		totalLoss := gorgonia.Must(gorgonia.ReduceAdd(losses))

		return Result{Output: x, Loss: totalLoss}, nil
	}
}
