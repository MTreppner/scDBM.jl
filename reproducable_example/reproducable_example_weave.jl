using Weave

cd("C:/Users/treppner/Dropbox/PhD/scDBM.jl/reproducable_example/")

# Weave Julia markdown file
Weave.weave("reproducable_example.jmd", out_path=:pwd)

# Weave IPython notebook
Weave.notebook("reproducable_example.jmd"; out_path=:pwd)
