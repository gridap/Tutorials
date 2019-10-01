sed 's/%.*$//g' $1 | \
sed '/^.*\\begin{figure}.*$/d' | \
sed '/^.*\\end{figure}.*$/d' | \
sed '/^.*\\begin{itemize}.*$/d' | \
sed '/^.*\\end{itemize}.*$/d' | \
sed '/^.*\\label.*$/d' | \
sed '/^.*\\caption.*$/d' | \
sed '/^.*\\begin{minted}.*$/d' | \
sed '/^.*\\end{minted}.*$/d' | \
sed 's/\\shb{\([^}]*\)}/`\1`/g' | \
sed 's/\\emph{\([^}]*\)}/*\1*/g' | \
sed 's/\\subsection{\([^}]*\)}/## \1/g' | \
sed 's/\\includegraphics\[width=0.6\\textwidth\]{\([^}]*\)}/![](\1)/g' | \
sed 's/\\gridap{}/Gridap/g' | \
sed 's/\\gridap/Gridap/g' | \
sed 's/\\julia{}/Julia/g' | \
sed 's/\\julia/Julia/g' | \
sed 's/\\paraview{}/Paraview/g' | \
sed 's/\\paraview/Paraview/g' | \
sed 's/\\fig{\([^}]*\)}/next figure/g' | \
sed 's/\\ac{fe}/FE/g' | \
sed 's/\\ac{pde}/PDE/g' | \
sed 's/\\acp{pde}/PDEs/g' | \
sed 's/\\begin{align}/```math/g' | \
sed 's/\\end{align}/```/g' | \
sed 's/\\begin{equation}/```math/g' | \
sed 's/\\end{equation}/```/g' | \
sed 's/\\linebreak//g' | \
sed 's/\\item/ -/g' | \
sed 's/\\_/_/g'
