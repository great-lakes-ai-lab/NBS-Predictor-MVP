# Install general Jupyter extensions
pip install jupyter jupyter_contrib_nbextensions
jupyter nbextensions_configurator enable --user
jupyter contrib nbextension install --user

# Install and enable jupyter-black
jupyter nbextension install https://github.com/drillan/jupyter-black/archive/master.zip --user
jupyter nbextension enable jupyter-black-master/jupyter-black

# Enable selected Jupyter Notebook extensions
for ext in \
    toggle_all_line_numbers/main \
    execute_time/ExecuteTime \
    codefolding/main \
    collapsible_headings/main \
    highlight_selected_word/main \
    move_selected_cells/main \
    scratchpad/main \
    autosavetime/main \
    varInspector/main \
    freeze/main \
    hide_input_all/main; do
    jupyter nbextension enable $ext
done
