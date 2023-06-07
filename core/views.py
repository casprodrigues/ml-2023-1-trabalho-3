import os
from django.views.generic import TemplateView
from django.conf import settings


class HomeView(TemplateView):
    template_name = 'home.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        image_names = [
            'acuracia.png',
            'curva_aprendizado.png',
            'distribuicao_pesos.png',
        ]

        media_url = settings.MEDIA_URL
        media_root = settings.MEDIA_ROOT

        image_urls = [os.path.join(media_url, name) for name in image_names]
        image_paths = [os.path.join(media_root, name) for name in image_names]

        context['image_urls'] = image_urls
        return context