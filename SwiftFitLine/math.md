
$line = f(x) = w \cdot x + b$

The loss function, Sum of Square Residuals:

$E = ∑(y_t - y_p)^2 = ∑(y_t - f(x))^2 = ∑(y_t - w \cdot x - b)^2$

$y_t$ - training value
$y_p$ - predicted value

The chain rule: $\frac{∂f}{∂x}g(f(x)) = g'(f(x)) \cdot f'(x)$


Gradients:

$∂E/∂w = ∑\frac{∂}{∂w}(y_t - w \cdot x - b)^2$
            $= ∑2(y_t - w \cdot x - b)(-x)$
            $= ∑-2x(y_t - w \cdot x - b)$

$∂E/∂b = ∑\frac{∂}{∂w}(y_t - w \cdot x - b)^2$
          $= ∑2(y_t - w \cdot x - b)(-1)$
          $= ∑-2(y_t - w \cdot x - b)$

$w_{new} = w - LearningRate \cdot (∂E/∂w)$
$b_{new} = b - LearningRate \cdot (∂E/∂b)$
