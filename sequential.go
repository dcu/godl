package deepzen

import "gorgonia.org/gorgonia"

// Sequential runs the given layers one after the other
func Sequential(m *Model, layers ...Layer) Layer {
	_ = AddLayer("Sequential")

	return func(inputs ...*gorgonia.Node) (Result, error) {
		losses := make([]*gorgonia.Node, 0, len(layers))

		var (
			result Result
			err    error
		)

		for _, layer := range layers {
			result, err = layer(inputs...)
			if err != nil {
				return Result{}, err
			}

			if result.Loss != nil {
				losses = append(losses, result.Loss)
			}

			inputs = append([]*gorgonia.Node{result.Output}, result.Nodes...)
		}

		if len(losses) == 0 {
			return result, nil
		}

		result.Loss = gorgonia.Must(gorgonia.ReduceAdd(losses))

		return result, nil
	}
}
