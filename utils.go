package tabnet

import (
	"fmt"
)

func anyNumberToFloat64(v interface{}) float64 {
	switch f := v.(type) {
	case float64:
		return f
	case int:
		return float64(f)
	case int64:
		return float64(f)
	}

	panic(fmt.Errorf("unsupported type: %T", v))
}

func mustBeGreaterThan(lt layerType, context string, v interface{}, base interface{}) {
	if anyNumberToFloat64(v) <= anyNumberToFloat64(base) {
		panic(fmt.Errorf("[%s] %s: %v must be greater than %v", lt, context, v, base))
	}
}
