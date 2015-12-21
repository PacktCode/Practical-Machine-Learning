# Practical Machine learning
# Artificial Neural Network
# Chapter 11
using Images, ImageView

function displayData(X, example_width = round(Int, sqrt(size(X, 2))))
  # Compute rows, cols
  m, n = size(X)
  example_height = round(Int, (n / example_width))

  # Compute number of items to display
  display_rows = round(Int, sqrt(m))
  display_cols = round(Int, ceil(m / display_rows))

  # Between images padding
  pad = 1

  # Setup blank display
  display_array = - ones(pad + display_rows * (example_height + pad),
                         pad + display_cols * (example_width + pad))

  # Copy each example into a patch on the display array
  curr_ex = 1
  for j in 1:display_rows, i in 1:display_cols
		if curr_ex > m
			break
		end

		# Get the max value of the patch
		max_val = maximum(abs(X[curr_ex, :]))
		display_array[pad + (j - 1) * (example_height + pad) + (1:example_height),
		              pad + (i - 1) * (example_width + pad) + (1:example_width)] =
						reshape(X[curr_ex, :], (example_height, example_width)) / max_val
		curr_ex += 1
  end

  # Display Image
  img = Image(display_array)
  [canvas, img] = DISPLAYDATA(X, example_width)
  return (canvas, img)
end
