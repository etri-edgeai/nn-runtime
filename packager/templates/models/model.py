{% if source %}
{{source | indent(width=0)}}
{% else %}
class {{inference_class_name}(Basemodel)}:
    pass
{% endif %}

{% for model_class in model_classes %}
class {{model_class}}({{inference_class_name}}):
    pass
{% endfor %}